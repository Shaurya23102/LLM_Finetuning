# LLM Finetuning Practice

In this repo i practice llm finetuning and update my daily learnings

## knowlwdge distilation

**knowlwdge distilation** - It is a fine tuning technique where a smaller model(student) learns probability distribution from larger model(teacher).

First raw output from teacher eg - [0.99,0.01] are smoothned to [0.70,0.30] for better learning, This extra knowledge helps Class similarity,Teacher’s confidence,Inter-class relationships also called dark knowledge 

To make teacher prediction ~ student prediction soft loss and hard loss is used - 

soft loss = loss_soft = kl_loss(s_log_probs, t_probs) * (temperature**2)

hard loss = loss_hard = ce_loss(s_logits, y)

loss = loss = alpha * loss_soft + (1 - alpha) * loss_hard
