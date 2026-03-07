In this repo i practice llm finetuning and update my daily learnings

## Knowlwdge distilation

**Knowlwdge distilation** - It is a fine tuning technique where a smaller model(student) learns probability distribution from larger model(teacher).

First raw output from teacher eg - [0.99,0.01] are smoothned to [0.70,0.30] for better learning, This extra knowledge helps Class similarity,Teacher’s confidence,Inter-class relationships also called dark knowledge eg = “Cat” is more similar to “Tiger” than “Car”

To make teacher prediction ~ student prediction soft loss and hard loss is used - 

soft loss = loss_soft = kl_loss(s_log_probs, t_probs) * (temperature**2)

hard loss = loss_hard = ce_loss(s_logits, y)

loss = loss = alpha * loss_soft + (1 - alpha) * loss_hard

## Quantization 

**Quantization** - It is a fine tuning tecnique where weights of a model are converted from float32 to int8/int4 which reduces the model size and latency while comprimising a little in accuracy 

It is of 2 types QAT and PQT

**QAT** - The model is trained with quantization in mind — it learns to handle quantization during training itself.(the model is trained on int 8 weights) acc drop of 1-2%
**PQT** - The model has already been trained, and now we are quantizing it.(the model is trained on float32 weights but for inference it we convert it to int8) acc drop of 1-5%
Type Quantized Parameters Calibration
Static PTQ Weights ✅+ Activations 
Dynamic PTQ Weights ✅only 
Normal or Partial PTQ = Weight only Quant
Static PTQ = Calibration(training of few data to get range of weights) = Activation + Weight Quant
Dynamic PTQ = No Calibration = Weights are pre-quantized (saved as INT8), but activations are quantized temporarily during inference time (on-the-fly), not permanently —no calibration 
> required." and model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') is used during training, we don’t actually convert tensors to int8 (because gradients need float precision).Instead, PyTorch inserts special modules (FakeQuantize) that simulate quantization effects forward pass = int8 and backwardpass = float32
