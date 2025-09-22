class TestInputCreator(unittest.TestCase):
    
    def setUp(self):
        # Setup consistent mock inputs for testing
        self.mock_args = InputArgs(
            max_num_seq=4,
            max_prefill_len=60,
            max_model_len=1024,
            sampler=MockSampler(),
            num_kv_heads=8,
            head_dim=128,
            num_layers=2,
            vocab_size=32000,
            min_prefill_len=1,
            phase='decode',
            block_size=16,
            num_blocks_override=4 * mock_cdiv(1024, 16) # 4 seqs * 64 blocks/seq = 256 blocks
        )
        self.mock_mesh = MockMesh(axis_names=('data',))
        self.mock_sharding = MockNamedSharding(self.mock_mesh, MockPartitionSpec())
        self.mock_rngs = MockRngs()
        
        # Create an instance for general setup checks
        self.creator = InputCreator(
            self.mock_args,
            self.mock_sharding,
            self.mock_rngs,
            self.mock_mesh
        )
        
    def test_setup_calculations(self):
        """Test the basic setup and calculated properties."""
        self.assertEqual(self.creator.input_args.block_size, 16)
        
        # max_blocks_per_req = cdiv(1024, 16) = 64
        self.assertEqual(self.creator.max_blocks_per_req, 64)
        
        # num_blocks = num_blocks_override = 256
        self.assertEqual(self.creator.num_blocks, 256)
        
        # KV caches should be created for 2 layers
        self.assertEqual(len(self.creator.kv_caches), 2)
        
        # Check the shape of the mock KV caches: [num_blocks, block_size, num_kv_heads, head_dim]
        expected_shape = (256, 16, 8, 128)
        self.assertTrue(all(k.shape == expected_shape and v.shape == expected_shape 
                            for k, v in self.creator.kv_caches))
                            
    def test_create_mock_block_table(self):
        """Test the creation and shape of the block table."""
        # Expected shape: [max_num_seq, blocks_per_req] = [4, 64]
        self.assertEqual(self.creator.block_table.shape, (4, 64))
        
        # Check the contents: should be sequential integers
        expected_content = np.arange(256).reshape(4, 64)
        npt.assert_array_equal(self.creator.block_table, expected_content)

    def test_create_input_decode_phase(self):
        """Test the ModelInputs object created in 'decode' phase."""
        self.mock_args.phase = 'decode'
        self.creator = InputCreator(self.mock_args, self.mock_sharding, self.mock_rngs, self.mock_mesh)
        
        model_inputs = self.creator.create_input(phase='decode')
        meta = model_inputs.attention_metadata
        
        # 1. Input IDs (one token per sequence)
        # Size: max_num_seq = 4
        self.assertEqual(model_inputs.input_ids.shape, (4,))
        
        # 2. Sequence Lengths
        # Size: max_num_seq = 4. Content: [10, 10, 10, 10] (from mock)
        self.assertEqual(meta.seq_lens.shape, (4,))
        npt.assert_array_equal(meta.seq_lens, np.full(4, 10, dtype=np.int32))
        
        # 3. Query Start Locations (should be [0, 1, 2, 3, 4] for decode phase)
        # Size: max_num_seq + 1 = 5
        self.assertEqual(meta.query_start_loc.shape, (5,))
        npt.assert_array_equal(meta.query_start_loc, np.arange(5, dtype=np.int32))
        
        # 4. Request Distribution (Decode: [4, 4, 4])
        self.assertEqual(meta.request_distribution.shape, (3,))
        npt.assert_array_equal(meta.request_distribution, np.array([4, 4, 4], dtype=np.int32))
        
        # 5. KV Write Indices Metadata (Complex check)
        # 4 sequences, each writing 1 token, each writing to 1 block. Total 4 slices.
        # Expected shape is [3, 4] after transpose (3 rows for [kv_cache_start, new_kv_start, slice_len])
        expected_num_slices = 4
        self.assertEqual(self.creator.kv_cache_write_indices.shape[1], 
                         get_padded_num_kv_cache_update_slices(4, 4, 16))
        
        # Check the actual (unpadded) content of the KV write indices
        # We mocked seq_lens as [10, 10, 10, 10]. Decode writes 1 new token at position 10.
        # For req 0: write to block 0 at offset 10.
        # kv_cache_start_indices: [10, 10, 10, 10] (offset 10 in block 0, 64, 128, 192) - ERROR IN MOCK. 
        # Block index for seq 0, pos 10 is block 0. Block start is 0.
        # Block index for seq 1, pos 10 is block 64. Block start is 1024.
        
        # Re-evaluating the decode logic in _mock_kv_write_indices:
        # Request 0: seq_len=10. slice_start=10, slice_end=11. Block: 10//16=0. (11-1)//16=0. block_len=1.
        # Start offset: 10 % 16 = 10. End offset: (11-1) % 16 + 1 = 11.
        
        # Slice 0: kv_cache_start=0*16+10=10. new_kv_start=0. slice_len=1.
        # Slice 1: kv_cache_start=64*16+10=1034. new_kv_start=1. slice_len=1.
        # ...
        
        # We test the number of slices for the unpadded array:
        unpadded_write_indices = self.creator._mock_kv_write_indices(meta.seq_lens.tolist(), ['decode'] * 4)
        self.assertEqual(unpadded_write_indices.shape, (4, 3))
        
        # [kv_cache_start_indices, new_kv_start_indices, slice_lens]
        expected_unpadded_writes = np.array([
            [0*16 + 10, 0, 1], # Seq 0, Block 0 (physical 0)
            [64*16 + 10, 1, 1], # Seq 1, Block 64 (physical 64)
            [128*16 + 10, 2, 1], # Seq 2, Block 128 (physical 128)
            [192*16 + 10, 3, 1], # Seq 3, Block 192 (physical 192)
        ], dtype=np.int32)
        
        # Check that the logic in _mock_kv_write_indices is correct for the core logic
        npt.assert_array_equal(unpadded_write_indices, expected_unpadded_writes)

    def test_create_input_prefill_phase(self):
        """Test the ModelInputs object created in 'prefill' phase."""
        self.mock_args.phase = 'prefill'
        self.mock_args.max_prefill_len = 60 # 60 tokens / 16 block size = 4 blocks per req (3 full + 1 partial)
        self.creator = InputCreator(self.mock_args, self.mock_sharding, self.mock_rngs, self.mock_mesh)
        
        model_inputs = self.creator.create_input(phase='prefill')
        meta = model_inputs.attention_metadata
        
        # 1. Input IDs (flattened tokens)
        # Size: max_num_seq * max_prefill_len = 4 * 60 = 240
        self.assertEqual(model_inputs.input_ids.shape, (240,))
        
        # 2. Sequence Lengths
        # Size: 4. Content: [60, 60, 60, 60]
        self.assertEqual(meta.seq_lens.shape, (4,))
        npt.assert_array_equal(meta.seq_lens, np.full(4, 60, dtype=np.int32))
        
        # 3. Query Start Locations (cumulative sum of seq_lens)
        # Size: 5. Content: [0, 60, 120, 180, 240]
        self.assertEqual(meta.query_start_loc.shape, (5,))
        npt.assert_array_equal(meta.query_start_loc, np.array([0, 60, 120, 180, 240], dtype=np.int32))
        
        # 4. Request Distribution (Prefill: [0, 0, 4])
        self.assertEqual(meta.request_distribution.shape, (3,))
        npt.assert_array_equal(meta.request_distribution, np.array([0, 0, 4], dtype=np.int32))
        
        # 5. KV Write Indices Metadata
        # 4 sequences, 60 tokens each. 60 / 16 = 3 full blocks + 1 partial (4 blocks total).
        # Total slices = 4 seqs * 4 blocks/seq = 16 slices.
        
        unpadded_write_indices = self.creator._mock_kv_write_indices(meta.seq_lens.tolist(), ['prefill'] * 4)
        self.assertEqual(unpadded_write_indices.shape, (16, 3))
        
        # Check first slice of Req 0 (tokens 0-15 -> block 0)
        # Expected: kv_cache_start=0*16+0=0, new_kv_start=0, slice_len=16
        npt.assert_array_equal(unpadded_write_indices[0], np.array([0, 0, 16], dtype=np.int32))
        
        # Check last slice of Req 0 (tokens 48-59 -> block 3. Block start=48)
        # Block index is 3 (physical block 3). slice_start=48%16=0. slice_end=(60-1)%16+1=12.
        # This slice is at index 3 (block_lens_cumsum[1]-1).
        # Expected: kv_cache_start=3*16+0=48, new_kv_start=48, slice_len=12
        npt.assert_array_equal(unpadded_write_indices[3], np.array([48, 48, 12], dtype=np.int32))
        
        # Check first slice of Req 1 (tokens 60-75 -> block 4 (physical block 64))
        # This slice is at index 4 (block_lens_cumsum[1]).
        # Expected: kv_cache_start=64*16+0=1024, new_kv_start=60, slice_len=16
        npt.assert_array_equal(unpadded_write_indices[4], np.array([1024, 60, 16], dtype=np.int32))
        
    def test_kv_write_indices_mixed_block_offsets(self):
        """Test the complex index mapping logic with non-zero start/end offsets."""
        # Setup for a single sequence with a write that crosses block boundaries partially.
        mock_seq_lens = [35] # Write 35 tokens
        
        # Block size 16. 35 tokens -> Blocks 0, 1, 2. Total 3 slices.
        # Block 0: pos 0-15 (16 tokens)
        # Block 1: pos 16-31 (16 tokens)
        # Block 2: pos 32-34 (3 tokens)
        
        # Start offset (Slice 0) = 0 % 16 = 0
        # End offset (Slice 2) = (35 - 1) % 16 + 1 = 3 + 1 = 4
        
        unpadded_write_indices = self.creator._mock_kv_write_indices(mock_seq_lens, ['prefill'])
        
        self.assertEqual(unpadded_write_indices.shape, (3, 3))
        
        # Block Indices used by Req 0: 0, 1, 2 (from self.block_table.flatten())
        
        # Slice 0 (Block 0): Full block write (0-15).
        # kv_cache_start: 0*16 + 0 = 0. new_kv_start: 0. slice_len: 16.
        npt.assert_array_equal(unpadded_write_indices[0], np.array([0, 0, 16], dtype=np.int32))
        
        # Slice 1 (Block 1): Full block write (16-31).
        # kv_cache_start: 1*16 + 0 = 16. new_kv_start: 16. slice_len: 16.
        npt.assert_array_equal(unpadded_write_indices[1], np.array([16, 16, 16], dtype=np.int32))
        
        # Slice 2 (Block 2): Partial block write (32-34). End offset is 4.
        # kv_cache_start: 2*16 + 0 = 32. new_kv_start: 32. slice_len: 4.
        npt.assert_array_equal(unpadded_write_indices[2], np.array([32, 32, 4], dtype=np.int32))

if __name__ == '__main__':
    # Run tests using the unittest framework
    unittest.main(argv=['first-arg-is-ignored'], exit=False)