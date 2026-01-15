from sklearn.model_selection import train_test_split

from mobility.spatial import StayPointDetector, LocationGenerator
from mobility.training import SMOTENC_OvO_Trainer, MarkovChain
from mobility.utils import DataLoader, get_logger, setup_logging, datetime_parser, input_generator


setup_logging(
    log_dir="../logs/mobility"
)

logger = get_logger(__name__)


if __name__=="__main__":

    data_path = "./data/002/Trajectory"
    
    reader = DataLoader(data_path)
    detector = StayPointDetector()
    generator = LocationGenerator()
    

    pfs = reader.load_user()

    sps, _ = detector.detect_staypoints(pfs)
    locs = generator.generate_locations(sps)

    data = datetime_parser(locs)

    loc_ids = data["origin_id"]
    chain = MarkovChain(loc_ids)

    diary = chain.generate_sequence(0, 50)
    logger.info(diary)

    # parsed = datetime_parser(locs)