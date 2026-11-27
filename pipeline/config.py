"""全局配置。密钥放 .secrets.json（已 gitignore），不硬编码进仓库。"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "图书语料库" / "markdown"
ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
PROMPTS = ROOT / "pipeline" / "prompts"

API_BASE = "https://api.yuuverne.ccwu.cc"
MODEL = "100milionseek/deepseek-v4.1-flash-free"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")

_secrets = ROOT / ".secrets.json"
API_KEY = json.loads(_secrets.read_text(encoding="utf-8"))["api_key"] if _secrets.exists() else ""

# 训练/资产用书（克拉克森本人）
TRAIN_BOOKS = [
    "克拉克森的农场2：我的牛又不见了",
    "克拉克森的农场3：我的猪会飞",
    "克拉克森的农场4：自食其果",
    "克拉克森的农场5：农夫之犬",
    "克拉克森的世界：Top Gear那些年（车评，1993.10-2011.12）",
    "克拉克森的世界：停不下来（车评，2003.6-2006.3）",
    "克拉克森的世界：全速出击（时评，2018.1-2019.12）",
    "克拉克森的世界：天生易怒（车评，Since 1999）",
    "克拉克森的世界：如我所说（时评，2013.3-2015.3）",
    "克拉克森的世界：年轻时代（车评，1985-1993）",
    "克拉克森的世界：心烦意乱（车评，2006.1-2007.12）",
    "克拉克森的世界：我的天啊！（时评，2006.1-2007.12）",
    "克拉克森的世界：我看世界（时评，2001.1-2003.12）",
    "克拉克森的世界：是真的吗？（车评，2015.1-2018.12）",
    "克拉克森的世界：机器，也有灵魂（车评，2000s）",
    "克拉克森的世界：汽车星球（车评，1995.1-1996.12）",
    "克拉克森的世界：言无不尽（时评，2015.3-2017.12）",
    "克拉克森的世界：转向疯狂（车评，2008.1-2009.12）",
    "克拉克森的世界：这有何难？（时评，2008.1-2009.12）",
    "克拉克森的世界：这能错吗？（车评，2011.1-2013.12）",
    "克拉克森的世界：问题多多（时评，2010.4-2013.3）",
    "克拉克森的世界：风云再起（时评，2004.1-2005.12）",
]

# 留出：不进入资产包，只用于测试/校准
HELDOUT_BOOKS = [
    "克拉克森的农场1：田园生活 从开始到放弃",
]

# 同域不同作者 → 判别器负样本
NEG_BOOKS = [
    "詹姆斯·梅文选第1卷 - Notes from the Hard Shoulder",
    "詹姆斯·梅文选第2卷 - Car Fever",
    "詹姆斯·梅文选第3卷 - May on Motors",
    "詹姆斯·梅文选第4卷 - Carbolics A personal motoring disinfectant",
    "詹姆斯·梅 - How to Land an A330 Airbus and Other Vital Skills for the Modern Man",
    "詹姆斯·梅 - James May's Man Lab",
    "理查德·哈蒙德自传 - On the Edge",
    "安迪·威尔曼的Top Gear、The Grand Tour汽车之旅",
    "卡勒布·库珀 - The World According to Kaleb",
    "格温·拉塞尔 - 杰里米·克拉克森传",
]

GENRE_MAP = {
    "车评": "car",
    "时评": "column",
    "农场": "farm",
}
