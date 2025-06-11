import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Model type:", cfg.model.type)
    print("Embedding dim:", cfg.model.lstm.embedding_dim)
    print("Optimizer:", cfg.optimizer.type)
    print("Learning rate:", cfg.optimizer.learning_rate)

if __name__ == "__main__":
    main()
