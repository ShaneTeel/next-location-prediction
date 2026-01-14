from mobility.utils import DataLoader, get_logger

logger = get_logger(__name__)


def main(data_path):
    reader = DataLoader(data_path=data_path)
    data = reader.load_user()

if __name__=="__main__":
    data_path = "./data/002/Trajectory"
    main(data_path)