def wanda_matrix_hook(inst, inp, out, layer, lin):
    """
    Captures the full Wanda Matrix (|W| * ||A||) for the layer.
    """
    with torch.no_grad():
        # W shape: [Out, In]
        W = inst.weight
        C_out = W.shape[1]

        # Calculate L2 norm of input activations across batch and tokens
        # inp[0] is [Batch, Tokens, In_Dim] -> Flatten to [-1, In_Dim]
        l2_norm = inp[0].reshape(-1, C_out).norm(p=2, dim=0) # [In_Dim]
        
        # Wanda Matrix: Broadcast multiply
        # [Out, In] * [In] -> [Out, In]
        wanda_mat = W.abs() * l2_norm
        
        # Accumulate
        if lin == 1:
            if 'W1' not in wanda_matrices[layer]:
                wanda_matrices[layer]['W1'] = torch.zeros_like(wanda_mat)
            wanda_matrices[layer]['W1'] += wanda_mat
        elif lin == 2:
            if 'W2' not in wanda_matrices[layer]:
                wanda_matrices[layer]['W2'] = torch.zeros_like(wanda_mat)
            wanda_matrices[layer]['W2'] += wanda_mat
            
        # Count batches (increment only once per layer pass, using lin=1)
        if lin == 1:
            wanda_matrices[layer]['count'] = wanda_matrices[layer].get('count', 0) + 1

def vit_wanda_graph_ricci_reordering(model, dataloader, sparse_threshold=0.01):
    """
    1. Compute Wanda Matrices (Activation-Aware Weights).
    2. Build Graph using Wanda Matrices (Traffic-based Topology).
    3. Compute Ricci Curvature on this Traffic Graph.
    4. Reorder based on Hybrid Score (Wanda_Mag - Ricci).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # --- PHASE 1: Calibration (Compute Wanda Matrices) ---
    print("Phase 1: Building Activation-Aware Graphs (Wanda Calibration)...")
    
    global wanda_matrices
    wanda_matrices = {i: {} for i in range(len(model.vit.encoder.layer))}
    
    hooks = []
    for idx, layer in enumerate(model.vit.encoder.layer):
        h1 = layer.intermediate.dense.register_forward_hook(partial(wanda_matrix_hook, layer=idx, lin=1))
        h2 = layer.output.dense.register_forward_hook(partial(wanda_matrix_hook, layer=idx, lin=2))
        hooks.extend([h1, h2])
        
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Forward Passes"):
            if isinstance(batch, dict):
                inputs = batch.get("pixel_values", batch.get("img")).to(device)
            else:
                inputs = batch[0].to(device)
            model(inputs)
            
    for h in hooks: h.remove()
    
    # --- PHASE 2 & 3: Geometric Analysis & Reordering ---
    print("Phase 2: Computing Forman-Ricci on Wanda Graphs...")
    
    score_dist = {}
    
    for i, layer in enumerate(tqdm(model.vit.encoder.layer, desc="Reordering")):
        # Retrieve averaged Wanda Matrices
        count = wanda_matrices[i]['count']
        # W1_wanda: [Hidden, Input]
        W1_wanda = (wanda_matrices[i]['W1'] / count).cpu().numpy()
        # W2_wanda: [Output, Hidden]
        W2_wanda = (wanda_matrices[i]['W2'] / count).cpu().numpy()
        
        D_hid, D_in = W1_wanda.shape
        D_out, _ = W2_wanda.shape
        
        # --- Build Graph based on WANDA SCORES ---
        G = nx.DiGraph()
        
        # Thresholding based on Wanda strength
        thresh_w1 = sparse_threshold * np.max(W1_wanda)
        thresh_w2 = sparse_threshold * np.max(W2_wanda)
        
        # Add Edges: Distance is inverse of Wanda Score
        # (High Traffic = Short Distance = Strong Connection)
        rows, cols = np.where(W1_wanda >= thresh_w1)
        for r, c in zip(rows, cols):
            w_val = W1_wanda[r, c]
            G.add_edge(c, D_in + r, weight=1.0/(w_val + 1e-6), layer='W1', hidden_idx=r)
            
        rows, cols = np.where(W2_wanda >= thresh_w2)
        for r, c in zip(rows, cols):
            w_val = W2_wanda[r, c]
            G.add_edge(D_in + c, D_in + D_hid + r, weight=1.0/(w_val + 1e-6), layer='W2', hidden_idx=c)
            
        # --- Compute Ricci ---
        ricci_neuron_scores = np.zeros(D_hid)
        
        if G.number_of_edges() > 0:
            try:
                # Using Forman-Ricci for speed
                frc = FormanRicciGPU(G,batch_size=2048, verbose="ERROR",device='cuda')
                frc.compute_ricci_curvature()
                
                for u, v, data in frc.G.edges(data=True):
                    if 'formanCurvature' in data:
                        h_idx = data.get('hidden_idx')
                        if h_idx is not None:
                            ricci_neuron_scores[h_idx] += data['formanCurvature']
            except Exception as e:
                print(f"Ricci Error Layer {i}: {e}")
        
        # --- Compute Node-Level Wanda Magnitude ---
        # (Total traffic passing through the neuron)
        w1_sum = np.sum(W1_wanda, axis=1) # [Hidden]
        w2_sum = np.sum(W2_wanda, axis=0) # [Hidden]
        wanda_mag = (w1_sum + w2_sum) / 2.0
        
        # --- Hybrid Combination ---
        # Normalize
        w_norm = (wanda_mag - wanda_mag.min()) / (wanda_mag.max() - wanda_mag.min() + 1e-6)
        
        max_r = np.max(np.abs(ricci_neuron_scores))
        if max_r > 0:
            r_norm = ricci_neuron_scores / max_r
        else:
            r_norm = ricci_neuron_scores
            
        # Score = Traffic_Strength - Traffic_Bottleneck
        # (Negative Ricci on Wanda Graph = Critical Traffic Bottleneck)
        hybrid_scores = - r_norm
        
        
        

        # score_dist[i] = hybrid_scores.tolist()
        
        # --- Reorder ---
        sorted_indices = np.argsort(hybrid_scores)[::-1]
        idx = torch.from_numpy(sorted_indices.copy()).long().to(device)
        
        # Apply to actual weights
        layer.intermediate.dense.weight.data = layer.intermediate.dense.weight.data[idx, :]
        layer.output.dense.weight.data = layer.output.dense.weight.data[:, idx]
        layer.intermediate.dense.bias.data = layer.intermediate.dense.bias.data[idx]
        
        # sort hybrid scores and store in score_dist
        score_dist[i] = torch.from_numpy(hybrid_scores[sorted_indices].copy()).tolist()

    return score_dist

# Global dictionary to store Node-Level Wanda scores
global hybrid_wanda_nodes
hybrid_wanda_nodes = {}

def hybrid_wanda_hook(inst, inp, out, layer, proj_name):
    """
    Captures the Node-Level Wanda Score for the layer.
    Instead of saving massive matrices, we immediately sum across the 
    output dimension to get the 1D importance score per input channel.
    """
    with torch.no_grad():
        W = inst.weight # [Out_Dim, In_Dim]
        In_Dim = W.shape[1]

        # Calculate L2 norm of input activations across batch and tokens
        l2_norm = inp[0].reshape(-1, In_Dim).norm(p=2, dim=0) # [In_Dim]
        
        # Wanda Matrix: [Out, In]
        wanda_mat = W.abs() * l2_norm
        
        # Node-level Traffic (Sum traffic leaving each input channel)
        node_traffic = wanda_mat.sum(dim=0) # [In_Dim]
        
        if layer not in hybrid_wanda_nodes:
            hybrid_wanda_nodes[layer] = {'count': 0}
            
        if proj_name not in hybrid_wanda_nodes[layer]:
            hybrid_wanda_nodes[layer][proj_name] = torch.zeros_like(node_traffic)
            
        hybrid_wanda_nodes[layer][proj_name] += node_traffic
        
        if proj_name == 'Q':
            hybrid_wanda_nodes[layer]['count'] += 1

def vit_hybrid_attention_pruning(model, dataloader, prune_ratio=0.3, sparsity_threshold=0.7, device="cuda"):
    """
    Hybrid Pipeline:
    1. Hook activations to compute Node-level Wanda traffic.
    2. Build graph using pure weights to compute pure geometric Ricci.
    3. Hybrid Score = Normalized(Wanda) - Normalized(Ricci).
    4. Prune channels with the lowest Hybrid Score.
    """
    model = model.to(device)
    model.eval()
    
    # --- PHASE 1: Calibration (Compute Node Wanda) ---
    print("\nPhase 1: Calibrating Activation Traffic (Wanda)...")
    
    global hybrid_wanda_nodes
    hybrid_wanda_nodes = {}
    hooks = []
    
    for idx, layer in enumerate(model.vit.encoder.layer):
        attn_module = layer.attention.attention
        hq = attn_module.query.register_forward_hook(partial(hybrid_wanda_hook, layer=idx, proj_name='Q'))
        hk = attn_module.key.register_forward_hook(partial(hybrid_wanda_hook, layer=idx, proj_name='K'))
        hooks.extend([hq, hk])
        
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Forward Passes"):
            if isinstance(batch, dict):
                inputs = batch.get("pixel_values", batch.get("img")).to(device)
            else:
                inputs = batch[0].to(device)
            model(inputs)
            
    for h in hooks: 
        h.remove()
    
    # --- PHASE 2: Geometric Analysis & Hybrid Pruning ---
    print("\nPhase 2: Computing Ricci Topology & Applying Hybrid Mask...")
    
    for i, layer in enumerate(tqdm(model.vit.encoder.layer, desc="Pruning Layers")):
        # 1. Get Pure Weights for Topology
        W_q_pure = layer.attention.attention.query.weight.data
        W_k_pure = layer.attention.attention.key.weight.data
        dim = W_q_pure.shape[1]
        
        # 2. Build Graph and Compute Ricci (Pure Geometry)
        G, nodes = build_bilinear_attention_graph(W_q_pure, W_k_pure, sparsity_threshold)
        
        ricci_scores = np.zeros(dim)
        if G.number_of_edges() > 0:
            try:
                frc = FormanRicciGPU(G, weight="weight", method="augmented", batch_size=1024, device=device)
                frc.compute_ricci_curvature()
                ricci_scores = np.array([frc.G.nodes[n].get("formanCurvature", 0.0) for n in nodes])
            except Exception as e:
                print(f"Ricci Error Layer {i}: {e}")
                
        # 3. Retrieve Node-Level Wanda (Traffic)
        count = hybrid_wanda_nodes[i]['count']
        w_q_node = hybrid_wanda_nodes[i]['Q'].cpu().numpy() / count
        w_k_node = hybrid_wanda_nodes[i]['K'].cpu().numpy() / count
        wanda_scores = (w_q_node + w_k_node) / 2.0
        
        # 4. Normalize Scores
        # Min-Max normalize Wanda
        w_norm = (wanda_scores - wanda_scores.min()) / (wanda_scores.max() - wanda_scores.min() + 1e-6)
        
        # Max-Abs normalize Ricci to preserve the negative/positive signs
        max_r = np.max(np.abs(ricci_scores))
        r_norm = ricci_scores / max_r if max_r > 0 else ricci_scores
        
        # 5. Calculate Hybrid Score
        # High Wanda (critical traffic) -> Positive contribution
        # High Positive Ricci (redundant loops) -> Negative contribution
        # High Negative Ricci (critical bridges) -> Positive contribution (since we subtract a negative)
        hybrid_scores = w_norm - r_norm
        
        # 6. Generate Mask & Prune
        num_to_prune = int(dim * prune_ratio)
        
        # Sort ascending (lowest scores are the most redundant / lowest traffic)
        channels_to_drop = np.argsort(hybrid_scores)[:num_to_prune].tolist()
        
        mask = torch.ones(dim, dtype=torch.bool).to(device)
        mask[channels_to_drop] = False
        
        # Apply to Q, K, V
        layer.attention.attention.query.weight.data[:, ~mask] = 0.0
        layer.attention.attention.key.weight.data[:, ~mask] = 0.0
        layer.attention.attention.value.weight.data[:, ~mask] = 0.0