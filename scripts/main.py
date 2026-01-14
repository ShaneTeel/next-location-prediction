from mobility.spatial import StayPointDetector, LocationGenerator
from mobility.utils import DataLoader, get_logger, setup_logging


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
    coords = locs[["precise_lat", "precise_lon"]]
    labels = locs[["cluster"]]

    

    print(locs.head())
