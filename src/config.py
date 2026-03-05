from src import *
from dataclasses import dataclass
from enum import Enum

class MODE(str, Enum):
    OPTIMIZE = "OPTIMIZATION"            # do optimization
    SIMULATE = "SIMULATION"               # just simulation, no backward
    
    SOTA_OPTIMIZE = "SOTA_OPT"
    SOTA_SIMULATE = "SOTA_SIM"

###########################################
## CHANGE HERE ##

setting_fileName =  "Scenario1"
mode = "SIMULATION"
lr = 0.07

###########################################

root = os.path.dirname(os.path.abspath(__file__))
Scene_folder = f"{root}/Scenarios/"
result_folder = f"{root}/Results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class Config:
    scenario: str = f"{Scene_folder}/{setting_fileName}.yaml"  # scenario file name
    run_mode: MODE = mode                                           # ruinning mode
    lr: float = lr                                              # learning rate
    control_vars: torch.tensor = torch.load(f"{Scene_folder}/{setting_fileName}.pt")
    # control_vars: torch.tensor = torch.from_numpy(np.loadtxt(str(Path(result_folder) / "bests" / "RLS1_best.txt")))
    current_time: str = time.strftime("%y%m%d_%H%M", time.localtime())


# save data
class Log:
    def __init__(self):
        self.log_buffer: list[str] = []
        self.folderName = ""
        self.logFolderName = ""

    def mf(self, current_time:str):
        self.folderName = f"{result_folder}/{current_time}"
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)

        return self.folderName
    
    def mlf(self):
        self.logFileName = f"{self.folderName}/log_{setting_fileName}.txt"
        self.logTxt = open(self.logFileName, "w")

    def log_flush(self):
        with open(self.logFileName, "a", encoding="utf-8") as f:
            f.write("\n".join(self.log_buffer) + "\n")
        self.log_buffer.clear()



cfg = Config()
log = Log()