from mobility.analyze import DataQualityAssessment
from mobility.utils import DataLoader, get_logger, setup_logging

setup_logging()

logger = get_logger(__name__)

if __name__=="__main__":
    reader = DataLoader("data/002/Trajectory", "Asia/Shanghai")

    pfs = reader.load_user()
    eval = DataQualityAssessment("002", pfs)
    logger.info(eval.generate_source_statement())