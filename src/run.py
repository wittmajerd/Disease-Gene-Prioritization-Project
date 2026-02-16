from __future__ import annotations

def main():
    cfg = load_config(...)
    seed_everything(cfg.seed)
    logger = setup_logging(cfg)
    
    raw = load_raw_data(cfg.graph.data_path)
    data = build_hetero_graph(raw, cfg.graph)
    splits = split_target_edges(data, cfg.graph.target_edge, cfg.split)
    
    model = HeteroLinkPredictor(data.metadata(), node_counts, cfg.model)
    trainer = Trainer(model, cfg, device, logger)
    
    train_result = trainer.fit(data, splits)
    test_result = trainer.test(data, splits.test)
    
    save_results(...)


if __name__ == "__main__":
    main()

    # python -m src.run_pipeline

