from mobility.utils import DataLoader, get_logger

logger = get_logger(__name__)


def main(data_path):
    data = DataLoader(data_path=data_path)
    logger.info("Loaded data sucessfull.")

if __name__=="__main__":
    data_path = "./data/002/Trajectory"
    main()