


class Coconut(nn.Module):
    """Meta's exact COCONUT implementation with multi-pass reasoning"""

    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # Meta's exact embedding extraction (tested with GPT2 and Llama3)
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        """Meta's exact multi-pass reasoning forward pass"""
        
        logits = []

        # Find all latent token positions in the batch
        latent_indices = (input_ids == self.latent_token_id).nonzero()  # (num_latent_tokens_in_batch, 2)

        # Group latent positions by batch item
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # Process before the earliest latent token position

        kv_cache = None

        # Meta's iterative multi-pass reasoning
        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # First forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # Reuse KV cache from previous pass
                past_key_values = [
                    (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :])
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            # Update compute range for next iteration
            next_compute_range = (
                next_compute_range[1],
                (input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1),
            )

            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            kv_cache = outputs.past_key_values

            # Meta's continuous thought feedback mechanism
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Break down embeddings to avoid in-place operations
            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace latent token embeddings with computed hidden states
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                # Use hidden state from position before the latent token
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # Reassemble the input embeddings
            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[batch_idx]) for batch_idx in range(inputs_embeds.shape[0])]
            )

        # Final forward pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache]
                if kv_cache else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1

        # Combine all logits and calculate loss
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(self, input_ids, attention_mask, max_new_tokens=16, output_embedding=False, synced_gpus=False, **kwargs):
        """Meta's exact generation method"""
        
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        # Process latent tokens with multi-pass reasoning
        labels = input_ids.clone()  # placeholder
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # Generate first new token
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Continue autoregressive generation
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(torch.tensor(next_token, device=input_ids.device)).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        # Handle distributed training synchronization
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)

print("✅ Meta's COCONUT class implemented")


def setup_base_model(model_id, vocab_size):
    """Setup GPT-2 model with resized embeddings"""
    print(f"🔧 Loading base model: {model_id}")
    
    # Load GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    print(f"   Original vocab size: {model.config.vocab_size}")
    print(f"   Target vocab size: {vocab_size}")
    
    # Resize token embeddings for new special tokens
    model.resize_token_embeddings(vocab_size)
    
    print(f"   Model resized to vocab size: {model.config.vocab_size}")
    return model

base_model = setup_base_model(cot_config.model_id, len(tokenizer))


def initialize_special_tokens(model, tokenizer, start_id, end_id, latent_id):
    """Meta's exact special token initialization using '<<' as template"""
    print("🔧 Initializing special tokens with Meta's exact method...")
    
    # Get embeddings and lm_head
    embeddings = model.get_input_embeddings()
    lm_head = model.lm_head
    
    # Meta uses '<<' token as initialization template
    target_token = "<<"
    target_id = tokenizer.convert_tokens_to_ids(target_token)
    
    if target_id == tokenizer.unk_token_id:
        print(f"   Warning: '{target_token}' not found, using random initialization")
        # Fallback to random initialization
        for token_id in [latent_id, start_id, end_id]:
            nn.init.normal_(embeddings.weight.data[token_id], mean=0.0, std=0.02)
            nn.init.normal_(lm_head.weight.data[token_id], mean=0.0, std=0.02)
    else:
        print(f"   Using '{target_token}' (ID: {target_id}) as initialization template")
        
        # Meta's exact initialization
        target_embedding = embeddings.weight.data[target_id]
        target_lm_weight = lm_head.weight.data[target_id]
        
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = target_embedding.clone()
            lm_head.weight.data[token_id] = target_lm_weight.clone()
    
    print(f"   ✅ Initialized tokens: {start_id}, {end_id}, {latent_id}")

initialize_special_tokens(base_model, tokenizer, start_id, end_id, latent_id)


def create_coconut_model(base_model, start_id, end_id, latent_id, eos_token_id):
    """Create Meta's COCONUT wrapper model"""
    print("🥥 Creating COCONUT wrapper model...")
    
    coconut_model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=eos_token_id
    )
    
    # Move to GPU
    coconut_model = coconut_model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in coconut_model.parameters())
    trainable_params = sum(p.numel() for p in coconut_model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model device: {next(coconut_model.parameters()).device}")
    
    return coconut_model

coconut_model = create_coconut_model(base_model, start_id, end_id, latent_id, tokenizer.eos_token_id)


def test_coconut_model(model, tokenizer, start_id, end_id, latent_id):
    """Test COCONUT model with sample inputs"""
    print("🧪 Testing COCONUT model...")
    
    # Create test input with latent tokens
    test_input = [50256] + [start_id] + [latent_id] * 4 + [end_id] + [220, 464]  # Simple test sequence
    
    input_ids = torch.tensor([test_input]).to(device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    position_ids = torch.arange(len(test_input)).unsqueeze(0).to(device)
    
    print(f"   Test input shape: {input_ids.shape}")
    print(f"   Latent tokens in input: {(input_ids == latent_id).sum().item()}")
    
    # Test forward pass
    model.train()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids
        )
    
    print(f"   ✅ Forward pass successful")
    print(f"   Output loss: {outputs.loss.item():.4f}")
    print(f"   Output logits shape: {outputs.logits.shape}")
    print(f"   Generation forward count: {model.gen_forward_cnt}")
    
    # Test generation
    model.eval()
    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids[:, :6],  # Just question + latent tokens
            attention_mask=attention_mask[:, :6],
            max_new_tokens=10
        )
    
    print(f"   ✅ Generation successful")
    print(f"   Generated sequence length: {gen_tokens.shape[1]}")
    
    return True

test_success = test_coconut_model(coconut_model, tokenizer, start_id, end_id, latent_id)

globals().update({
    # Models
    'base_model': base_model,
    'coconut_model': coconut_model,
    'Coconut': Coconut,
    
    # Setup functions
    'setup_base_model': setup_base_model,
    'initialize_special_tokens': initialize_special_tokens,
    'create_coconut_model': create_coconut_model,
    'test_coconut_model': test_coconut_model,
})

