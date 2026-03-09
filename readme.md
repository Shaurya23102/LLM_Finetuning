In this repo i practice llm finetuning and update my daily learnings

## Knowlwdge distilation

**Knowlwdge distilation** - It is a fine tuning technique where a smaller model(student) learns probability distribution from larger model(teacher).

First raw output from teacher eg - [0.99,0.01] are smoothned to [0.70,0.30] for better learning, This extra knowledge helps Class similarity,Teacher’s confidence,Inter-class relationships also called dark knowledge eg = “Cat” is more similar to “Tiger” than “Car”

To make teacher prediction ~ student prediction soft loss and hard loss is used - 

soft loss = loss_soft = kl_loss(s_log_probs, t_probs) * (temperature**2)

hard loss = loss_hard = ce_loss(s_logits, y)

loss = loss = alpha * loss_soft + (1 - alpha) * loss_hard

## Quantization

**Quantization** – It is a fine-tuning technique where weights of a model are converted from float32 to int8/int4, which reduces the model size and latency while compromising a little in accuracy.

It is of 2 types: QAT and PQT.

**QAT** – The model is trained with quantization in mind — it learns to handle quantization during training itself.  
(The model is trained on int8 weights).

If `model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')` is used during training, we don’t actually convert tensors to int8 (because gradients need float precision for back propogation). Instead, PyTorch inserts special modules called **FakeQuantize** that simulate quantization effects.

Forward pass = int8 (simulated)  
Backward pass = float32

Accuracy drop: 1–2%

**PQT** – The model has already been trained, and now we are quantizing it.  
(The model is trained on float32 weights but for inference we convert it to int8)  
Accuracy drop: 1–5%

Type | Quantized Parameters | Calibration
--- | --- | ---
Static PTQ | Weights + Activations | Required
Dynamic PTQ | Weights only | Not required
Normal / Partial PTQ | Weight-only quantization | Not required

**Static PTQ**  
Calibration (training on a few data samples to get the range of weights) is required.  
Activation + Weight Quantization.

**Dynamic PTQ**  
No calibration required.  
Weights are pre-quantized (saved as INT8), but activations are quantized temporarily during inference time (on-the-fly), not permanently.

