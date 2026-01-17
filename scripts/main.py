
from mobility.profile import StayPointDetector, LocationGenerator
from mobility.simulate import TimeAwareMarkov
from mobility.utils import DataLoader, get_logger, setup_logging


setup_logging(
    log_dir="../logs/mobility"
)

logger = get_logger(__name__)


if __name__=="__main__":

    data_path = "./data/000/Trajectory"
    
    reader = DataLoader(data_path)
    detector = StayPointDetector()
    generator = LocationGenerator()
    Tchain = TimeAwareMarkov()
    

    pfs = reader.load_user()

    sps, _ = detector.detect_staypoints(pfs)
    locs = generator.generate_locations(sps)
    labels = locs["origin_id"]
    datetime = locs["arrived"]

    predictions = Tchain.fit_predict(labels, datetime, start=0)

    print(predictions)