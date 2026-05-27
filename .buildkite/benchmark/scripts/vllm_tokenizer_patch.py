import sys
import runpy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vLLM-Hot-Patch")

logger.info("=====================================================")
logger.info("🚨 INJECTING VLLM DATASET HOT-PATCH")
logger.info("🚨 OVERRIDING CustomDataset.sample() FOR RAW IDS")
logger.info("=====================================================")

try:
    import vllm.benchmarks.datasets
    from vllm.benchmarks.datasets import SampleRequest

    # ====================================================================
    # Our malicious patched version of the sample function
    # ====================================================================
    def patched_sample(self, tokenizer, num_requests, request_id_prefix="", no_oversample=False, **kwargs):
        self.num_available_samples = len(self.data)
        if num_requests <= 0:
            num_requests = self.num_available_samples

        sampled_requests = []
        for i, item in enumerate(self.data):
            if len(sampled_requests) >= num_requests:
                break
            
            prompt = item["prompt"]
            output_len = kwargs.get("output_len")
            new_output_len = output_len if output_len is not None else int(item.get("output_tokens", 1024))
            
            # --- HOT PATCH CORE LOGIC ---
            if isinstance(prompt, list):
                # If prompt is a list (our raw token IDs), bypass the tokenizer entirely!
                prompt_len = len(prompt)
                sampled_requests.append(
                    SampleRequest(
                        prompt=prompt, # Passes the LIST natively to the request payload
                        prompt_len=prompt_len,
                        expected_output_len=new_output_len,
                        request_id=request_id_prefix + str(i),
                    )
                )
            else:
                # Fallback to vLLM's original string logic for normal benchmarks
                if not kwargs.get("skip_chat_template", False) and tokenizer is not None:
                    try:
                        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
                    except Exception:
                        pass
                prompt_len = len(tokenizer(prompt).input_ids) if tokenizer else 1
                sampled_requests.append(
                    SampleRequest(
                        prompt=prompt,
                        prompt_len=prompt_len,
                        expected_output_len=new_output_len,
                        request_id=request_id_prefix + str(i),
                    )
                )

        self.maybe_oversample_requests(sampled_requests, num_requests, request_id_prefix, no_oversample)
        return sampled_requests

    # Perform the monkey-patch
    vllm.benchmarks.datasets.CustomDataset.sample = patched_sample
    logger.info("✅ Successfully monkey-patched CustomDataset.sample!")
except Exception as e:
    logger.error(f"❌ Failed to patch vLLM: {e}")

if __name__ == "__main__":
    import shutil
    
    sys.argv.pop(0)
    
    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        logger.error("❌ Could not find 'vllm' executable in PATH. Make sure vLLM is installed.")
        sys.exit(1)

    sys.argv.insert(0, vllm_bin)
    
    runpy.run_path(vllm_bin, run_name="__main__")