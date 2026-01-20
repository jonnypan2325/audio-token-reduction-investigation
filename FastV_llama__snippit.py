# https://github.com/pkunlp-icler/FastV/blob/main/src/transformers/src/transformers/models/llama/modeling_llama.py

# FastV Token Rerank, Attention Mask Implementation
                elif USE_FAST_V:
                    if idx<AGG_LAYER:
                        new_attention_mask = torch.ones(
                            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                        )
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            new_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                        )
                        
                    elif idx==AGG_LAYER:
                        if idx!=0:
                            last_layer_attention = layer_outputs[1]
                            # compute average attention over different head
                            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                            # generate new attention mask based on the average attention, sample the top ATTENTION_RANK tokens with highest attention
                            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                            # get the attention in image token
                            last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                            # get the indexs of the top ATTENTION_RANK tokens
                            top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
                            # generate new attention mask
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
                            gen_attention_mask[:,SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = False
                            gen_attention_mask[:,top_attention_rank_index] = True

                            gen_attention_mask = self._prepare_decoder_attention_mask(
                                gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                            )
                            new_attention_mask = gen_attention_mask
                        
                        else:
                            gen_attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)

                            rand_image_attention_mask = [1]*ATTENTION_RANK + [0]*(IMAGE_TOKEN_LENGTH-ATTENTION_RANK)
                            random.shuffle(rand_image_attention_mask)

                            gen_attention_mask[:, SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH] = torch.tensor(rand_image_attention_mask, dtype=attention_mask.dtype, device=inputs_embeds.device)
                            gen_attention_mask = self._prepare_decoder_attention_mask(
                                gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                            )
                            new_attention_mask = gen_attention_mask