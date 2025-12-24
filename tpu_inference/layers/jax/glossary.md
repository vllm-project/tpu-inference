### Variable Glossary

| Variable | Full Name | Description |
| :--- | :--- | :--- |
| **B** | Batch Size | The number of samples processed at once. |
| **T** | Sequence Length | The number of tokens in the Query sequence. |
| **S** | Sequence Length | The number of tokens in the Key/Value sequence. |
| **D** | $d_{model}$ | The embedding dimension of the model. |
| **F** | $d_{ff}$ | The hidden dimension of the feed-forward MLP layers. |
| **V** | Vocab Size | The size of the vocabulary. |
| **H** | Head Dimension | The dimension of each attention head, typically $D/N$. |
| **N** | Number of Query Heads | The total number of query heads in multi-head attention. |
| **Q** | Number of Query Heads | Synonymous with **N**. |
| **K** | Number of Key/Value Heads | The total number of key/value heads. |
| **C** | Expert Capacity | The maximum number of tokens an expert can process in an MoE layer. |
| **X** | Activated Experts | The number of activated experts per token in MoE. |
| **G** | Number of Groups | The number of groups for grouped-query attention. |
| **E** | Total Experts | The total number of experts in the MoE layer. |
| **M** | Experts per Group | The number of experts within each group, where $M = E/G$. |
| **A** | Q Lora Rank | Used for DeepSeek models.
| **L** | Product of QK NoPE Head Dim and V Head Dim | Used for DeepSeek models.
| **P** | Product of Total (NoPE + RoPE) QK Head Dim and V Head Dim | Used for DeepSeek models.
| **R** | Product of Number of Attention Heads and V Head Dim | Used for DeepSeek models.
