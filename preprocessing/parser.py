import logging

from preprocessing import CodeBlock, LinkBlock, NormalTextBlock, TextSize
from lxml import etree

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DefaultParserInterface:
    def __init__(self):
        self.text_blocks = []

    def parse(self, data: str):
        text_block = NormalTextBlock(text=data, block_id=0, text_size=TextSize.P)
        self.text_blocks = [text_block]

        return self.text_blocks


class HTMLParserInterface:
    def __init__(self):
        self.parser = etree.HTMLParser()
        self.root = etree.fromstring("<html></html>", self.parser)
        self.text_blocks = []

    def __getstate__(self) -> object:
        data = self.__dict__.copy()
        del data["parser"]
        del data["root"]

        return data

    def __setstate__(self, state: object):
        self.__dict__.update(state)
        self.parser = etree.HTMLParser()
        self.root = etree.fromstring("<html></html>", self.parser)

    def feed(self, data: str):
        self.parser.feed(data)

    def parse(self, data: str):
        self.root = etree.fromstring(data, self.parser)
        self.text_blocks, _ = self.process_element(self.root)
        self.text_blocks = sorted(self.text_blocks, key=lambda x: x.block_id)

        return self.text_blocks

    def process_element(
        self, element: etree.Element, parent_element=None, parent_block=None, id=0
    ):
        if element is None:
            return [], None

        element_result, tailing_text = self._handle_element(
            element, parent_element=parent_element, parent_block=parent_block, id=id
        )

        element_results = []

        child_id = id
        for child in element:
            child_id += 1

            child_element_result, child_tailing_text = self.process_element(
                child, parent_element=element, parent_block=element_result, id=child_id
            )
            if child_element_result is not None:
                element_results.extend(child_element_result)

            child_id = (
                element_results[-1].block_id if len(element_results) > 0 else child_id
            )

            if child_tailing_text is not None and element_result is not None:
                child_id += 1
                secondary_text_block, _ = self._handle_tag(
                    tag=element.tag,
                    text=child_tailing_text,
                    id=child_id,
                    parent_tag=(
                        parent_element.tag if parent_element is not None else ""
                    ),
                    parent_block=parent_block,
                )
                if secondary_text_block is not None:
                    element_results.append(secondary_text_block)

        if element_result is not None:
            element_results.insert(0, element_result)

        return element_results, tailing_text

    def get_data(self) -> str:
        return etree.tostring(self.root, pretty_print=True)

    def _handle_element(
        self, element: etree.Element, parent_element=None, parent_block=None, **kwargs
    ):
        return self._handle_tag(
            tag=element.tag,
            text=element.text,
            tail=element.tail,
            parent_tag=parent_element.tag if parent_element is not None else "",
            parent_block=parent_block,
            attrib=element.attrib,
            **kwargs,
        )

    def _handle_tag(
        self,
        tag: str = "p",
        text: str = "",
        tail: str = "",
        parent_tag: str = "",
        parent_block=None,
        id: int = -1,
        **kwargs,
    ):
        if tag == "code":
            if parent_tag == "pre":
                text_block = CodeBlock(
                    text=text,
                    block_id=id,
                    in_line=False,
                ).update(parent_block)
            else:
                text_block = CodeBlock(
                    text=text,
                    block_id=id,
                    in_line=True,
                ).update(parent_block)
        elif tag in ["p", "h1", "h2", "h3", "h4", "h5", "h6"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                text_size=self._get_text_size(tag),
            ).update(parent_block)
        elif tag == "a":
            text_block = LinkBlock(
                text=text,
                block_id=id,
                href=kwargs.get("attrib", {}).get("href", ""),
                alt_text=kwargs.get("attrib", {}).get("alt", ""),
                text_size=self._get_text_size(parent_tag),
            ).update(parent_block)
        elif tag in ["strong", "b"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_bold=True,
            ).update(parent_block)
        elif tag in ["em", "i"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_italic=True,
            ).update(parent_block)
        elif tag == "u":
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_underline=True,
            ).update(parent_block)
        elif tag == "sup":
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_superscript=True,
            ).update(parent_block)
        elif tag in ["s", "del"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_strike_through=True,
            ).update(parent_block)
        elif tag in ["li", "ul", "ol"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_list=True,
            ).update(parent_block)
        elif tag == "blockquote":
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_blockquote=True,
            ).update(parent_block)
        elif tag in ["ul", "ol", "li"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_list=True,
            ).update(parent_block)
        elif text in ["dd", "dt", "dl"]:
            text_block = NormalTextBlock(
                text=text,
                block_id=id,
                is_desciption_list=True,
            ).update(parent_block)
        else:
            return None, None

        return text_block, tail

    def _get_parent(self, element: etree.Element):
        parent = element.getparent()
        if parent is None:
            return None

        return parent

    def _get_text_size(self, tag: str):
        upper_tag = tag.upper()
        if upper_tag in TextSize.__members__:
            return TextSize[upper_tag]

        return TextSize.UNK


if __name__ == "__main__":
    import argparse
    import pprint

    from constants.db import DB_PARAMS
    from utils.db_connection import DBConnection

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--use-test-data", action="store_true")
    args = parser.parse_args()

    parser = HTMLParserInterface()
    db_connection = DBConnection(DB_PARAMS)

    with open(".cache/test_documents.txt", "w") as f:
        f.write("id\ttitle\tbody\n")

    if not args.use_test_data:
        with db_connection as conn:
            select_query = """SELECT id, title, body FROM posts WHERE ID=1490039 OR ID=1308079 OR ID=631788 OR ID=1689012 OR ID=1449620 OR ID=1689145 OR ID=768941 OR ID=1115313 OR ID=1772491 OR ID=550585 OR ID=550915 OR ID=768956 OR ID=1529527 OR ID=1689139 OR ID=999182 OR ID=1728697 OR ID=550868 OR ID=1223197 OR ID=1601656 OR ID=506956 OR ID=1646326 OR ID=1264818 OR ID=1385753 OR ID=772220 OR ID=745600 OR ID=1520897 OR ID=1593100 OR ID=788935 OR ID=550829 OR ID=598776 OR ID=1690080 OR ID=951974 OR ID=646652 OR ID=1442686 OR ID=1382252 OR ID=550474 OR ID=784584 OR ID=1036512 OR ID=930183 OR ID=1633691 OR ID=1548620 OR ID=1442675 OR ID=1359557 OR ID=615129 OR ID=885158 OR ID=898326 OR ID=533768 OR ID=717655 OR ID=817937 OR ID=1308469 OR ID=1345786 OR ID=1264818 OR ID=1733292 OR ID=1184789 OR ID=1366832 OR ID=767009 OR ID=1028892 OR ID=1607800 OR ID=1697522 OR ID=961538 OR ID=1125604 OR ID=1580091 OR ID=1593100 OR ID=609329 OR ID=1259460 OR ID=580511 OR ID=1514112 OR ID=1533812 OR ID=841292 OR ID=1711103 OR ID=889924 OR ID=593397 OR ID=688002 OR ID=998049 OR ID=1208295 OR ID=582173 OR ID=989832 OR ID=966977 OR ID=1578347 OR ID=1790201 OR ID=1357321 OR ID=742549 OR ID=599469 OR ID=1150119 OR ID=1213489 OR ID=1738434 OR ID=1390571 OR ID=1027293 OR ID=852419 OR ID=1251464 OR ID=1347916 OR ID=960382 OR ID=1119373 OR ID=1649067 OR ID=1657432 OR ID=1691161 OR ID=1209370 OR ID=580448 OR ID=1406660 OR ID=1436994 OR ID=818020 OR ID=816834 OR ID=1731441 OR ID=1477365 OR ID=1790201 OR ID=742549 OR ID=1251464 OR ID=960382 OR ID=1119373 OR ID=1649067 OR ID=868899 OR ID=1252791 OR ID=1406660 OR ID=1536440 OR ID=730005 OR ID=1529527 OR ID=823171 OR ID=1296726 OR ID=729985 OR ID=1536275 OR ID=730064 OR ID=823764 OR ID=1369956 OR ID=627924 OR ID=487315 OR ID=1241723 OR ID=730157 OR ID=1366798 OR ID=870318 OR ID=818002 OR ID=651261 OR ID=648109 OR ID=1452953 OR ID=870127 OR ID=1313230 OR ID=1325275 OR ID=908330 OR ID=1473053 OR ID=763905 OR ID=1296772 OR ID=1153160 OR ID=610389 OR ID=921465 OR ID=857064 OR ID=887197 OR ID=1769480 OR ID=1238258 OR ID=857089 OR ID=870391 OR ID=818014 OR ID=1452239 OR ID=730064 OR ID=756384 OR ID=937491 OR ID=1256311 OR ID=971177 OR ID=1111227 OR ID=829798 OR ID=829916 OR ID=491999 OR ID=1432968 OR ID=826948 OR ID=1227605 OR ID=696889 OR ID=584284 OR ID=614993 OR ID=1776176 OR ID=659061 OR ID=1396782 OR ID=1291502 OR ID=1661491 OR ID=624492 OR ID=1439123 OR ID=609675 OR ID=1281654 OR ID=1719624 OR ID=1179892 OR ID=1077347 OR ID=1537931 OR ID=1732030 OR ID=1719928 OR ID=1582746 OR ID=1102673 OR ID=1114027 OR ID=1763783 OR ID=1731943 OR ID=1618739 OR ID=1574530 OR ID=1318787 OR ID=1421194 OR ID=629875 OR ID=881194 OR ID=1194913 OR ID=1236141 OR ID=1102752 OR ID=912466 OR ID=1432173 OR ID=614971 OR ID=1054100 OR ID=1615090 OR ID=1658569 OR ID=1432429 OR ID=1567015 OR ID=694369 OR ID=820054 OR ID=1054626 OR ID=590393 OR ID=754448 OR ID=844822 OR ID=1432093 OR ID=862624 OR ID=1432024 OR ID=1636119 OR ID=1383231 OR ID=1403484 OR ID=827647 OR ID=1657167 OR ID=1657248 OR ID=1434787 OR ID=1150723 OR ID=1083028 OR ID=685861 OR ID=1441880 OR ID=1657174 OR ID=998520 OR ID=846862 OR ID=1054605 OR ID=901098 OR ID=1260901 OR ID=1408318 OR ID=1441186 OR ID=754452 OR ID=703931 OR ID=1635807 OR ID=1536688 OR ID=1284074 OR ID=565971 OR ID=1692967 OR ID=1645191 OR ID=1257646 OR ID=1218241 OR ID=789885 OR ID=1698916 OR ID=1312584 OR ID=935019 OR ID=1062231 OR ID=810931 OR ID=1603479 OR ID=1252969 OR ID=565590 OR ID=829525 OR ID=1268725 OR ID=823764 OR ID=1790201 OR ID=1595849 OR ID=1452953 OR ID=765129 OR ID=630056 OR ID=1099837 OR ID=1251706 OR ID=1050889 OR ID=742549 OR ID=896840 OR ID=827121 OR ID=579255 OR ID=1251464 OR ID=1571771 OR ID=709949 OR ID=960382 OR ID=1366832 OR ID=1119373 OR ID=1649067 OR ID=788780 OR ID=819367 OR ID=1406660 OR ID=1168068 OR ID=893089 OR ID=1536850 OR ID=1529527 OR ID=845243 OR ID=1234550 OR ID=654093 OR ID=778837 OR ID=627924 OR ID=1272994 OR ID=1387983 OR ID=1572075 OR ID=1325275 OR ID=604836 OR ID=633289 OR ID=1309138 OR ID=1458973 OR ID=887197 OR ID=1769480 OR ID=1238258 OR ID=1425760 OR ID=1133434 OR ID=578533 OR ID=1345786 OR ID=1124670 OR ID=1309138 OR ID=846862 OR ID=829525 OR ID=1218241 OR ID=1153609 OR ID=715807 OR ID=1698916 OR ID=810931 OR ID=841713 OR ID=1595849 OR ID=765129 OR ID=630056 OR ID=887969 OR ID=1099837 OR ID=1297563 OR ID=1335257 OR ID=890461 OR ID=1183155 OR ID=1097093 OR ID=1106789 OR ID=1199490 OR ID=1246814 OR ID=1427023 OR ID=1710201 OR ID=1183639 OR ID=896840 OR ID=795914 OR ID=1148238 OR ID=785935 OR ID=838364 OR ID=624108 OR ID=1206811 OR ID=827121 OR ID=1307149 OR ID=1654196 OR ID=1182780 OR ID=1178585 OR ID=1756217 OR ID=709949 OR ID=993666 OR ID=1078972 OR ID=1239193 OR ID=1097080 OR ID=1126083 OR ID=1105740 OR ID=1168068 OR ID=1331992 OR ID=1570025 OR ID=536964 OR ID=1149150 OR ID=740943 OR ID=927179 OR ID=1088350 OR ID=1263783 OR ID=585004 OR ID=824002 OR ID=1410832 OR ID=1774106 OR ID=1351664 OR ID=1611961 OR ID=1546960 OR ID=859200 OR ID=632159 OR ID=1786635 OR ID=846862 OR ID=1587348 OR ID=1072548 OR ID=597664 OR ID=1109611 OR ID=1253861 OR ID=492411 OR ID=805201 OR ID=600243 OR ID=1150111 OR ID=1465247 OR ID=895648 OR ID=1637717 OR ID=1238258 OR ID=1092992 OR ID=582729 OR ID=797350 OR ID=782881 OR ID=1732510 OR ID=1621229 OR ID=797606 OR ID=1327742 OR ID=824081 OR ID=1482291 OR ID=1388696 OR ID=1512688 OR ID=1218241 OR ID=701711 OR ID=1725989 OR ID=1420677 OR ID=1698916 OR ID=930278 OR ID=575759 OR ID=1263625 OR ID=1547004 OR ID=773893 OR ID=1724137 OR ID=1547201 OR ID=1547196 OR ID=1724304 OR ID=1271643 OR ID=1516174 OR ID=1547210 OR ID=1285655 OR ID=1529201 OR ID=1289174 OR ID=1091524 OR ID=1289099 OR ID=1547224 OR ID=562770 OR ID=1289135 OR ID=1736340 OR ID=1737786 OR ID=1288985 OR ID=1737763 OR ID=1288904 OR ID=1697816 OR ID=1192007 OR ID=654010 OR ID=1543722 OR ID=1289139 OR ID=1723877 OR ID=608284 OR ID=674477 OR ID=712325 OR ID=450437 OR ID=716481 OR ID=1289152 OR ID=1370999 OR ID=1530487 OR ID=1508979 OR ID=578567 OR ID=1529267 OR ID=1551407 OR ID=1705857 OR ID=502392 OR ID=1247741 OR ID=608254 OR ID=928484 OR ID=609110 OR ID=1284259 OR ID=1563333 OR ID=1529219 OR ID=756296 OR ID=1425170 OR ID=646151 OR ID=994671 OR ID=631788 OR ID=533768 OR ID=1115313 OR ID=1529527 OR ID=1728697 OR ID=1504378 OR ID=749796 OR ID=1547019 OR ID=1067806 OR ID=1646326 OR ID=1264818 OR ID=1027739 OR ID=1623039 OR ID=745600 OR ID=1520897 OR ID=1593100 OR ID=768634 OR ID=566875 OR ID=951974 OR ID=646652 OR ID=1442686 OR ID=1688953 OR ID=1420925 OR ID=784584 OR ID=1156501 OR ID=596762 OR ID=954950 OR ID=971629 OR ID=1547782 OR ID=1442675 OR ID=653269 OR ID=885158 OR ID=1461942 OR ID=1471987 OR ID=1507041 OR ID=1035392 OR ID=1403134 OR ID=1545655 OR ID=1471945 OR ID=1684168 OR ID=1418266 OR ID=842570 OR ID=1671714 OR ID=1711578 OR ID=1416152 OR ID=1496118 OR ID=576664 OR ID=1014645 OR ID=1249744 OR ID=829525 OR ID=1307149 OR ID=1246261 OR ID=1771987 OR ID=1126083 OR ID=1595849 OR ID=765129 OR ID=630056 OR ID=1099837 OR ID=1387453 OR ID=1113029 OR ID=896840 OR ID=827121 OR ID=709949 OR ID=842467 OR ID=1168068 OR ID=791180 OR ID=893089 OR ID=1536850 OR ID=845243 OR ID=1268725 OR ID=1226004 OR ID=1234550 OR ID=1240306 OR ID=654093 OR ID=1758872 OR ID=1558285 OR ID=1636839 OR ID=1392173 OR ID=1641167 OR ID=1476052 OR ID=946948 OR ID=1164108 OR ID=1058348 OR ID=566292 OR ID=1255189 OR ID=1501638 OR ID=1572075 OR ID=1544006 OR ID=1188824 OR ID=1768799 OR ID=1117186 OR ID=681677 OR ID=1035091 OR ID=1163441 OR ID=693067 OR ID=1300804 OR ID=1758779 OR ID=1208070 OR ID=1173134 OR ID=786163 OR ID=1251706 OR ID=786235 OR ID=585363 OR ID=988763 OR ID=765054 OR ID=1790201 OR ID=742549 OR ID=1217479 OR ID=1251464 OR ID=960382 OR ID=1512688 OR ID=1251676 OR ID=1119373 OR ID=1649067 OR ID=1406660 OR ID=1268185 OR ID=1529527 OR ID=1346607 OR ID=780354 OR ID=773562 OR ID=823764 OR ID=1335326 OR ID=627924 OR ID=545432 OR ID=1452953 OR ID=1325275 OR ID=1031204 OR ID=617988 OR ID=887197 OR ID=1769480 OR ID=1238258 OR ID=1133434 OR ID=578533 OR ID=1366832 OR ID=1345786 OR ID=1124670 OR ID=1649029 OR ID=1132105 OR ID=861141 OR ID=715003 OR ID=1050889 OR ID=1728231 OR ID=1186028 OR ID=1529973 OR ID=640983 OR ID=569315 OR ID=1519597 OR ID=816834 OR ID=494720 OR ID=545003 OR ID=721797 OR ID=1275665 OR ID=1275799 OR ID=700241 OR ID=1233667 OR ID=1602944 OR ID=1727208 OR ID=750433 OR ID=1505510 OR ID=1727364 OR ID=750416 OR ID=1632209 OR ID=904641 OR ID=700263 OR ID=1374117 OR ID=796517 OR ID=1550970 OR ID=1744188 OR ID=868000 OR ID=750415 OR ID=1330350 OR ID=966914 OR ID=904654 OR ID=1055732 OR ID=1161330 OR ID=750313 OR ID=1070754 OR ID=1439922 OR ID=1116416 OR ID=1146729 OR ID=1705887 OR ID=842888 OR ID=1515042 OR ID=1650554 OR ID=750420 OR ID=842251 OR ID=821998 OR ID=1197026 OR ID=1046400 OR ID=700313 OR ID=750430 OR ID=750450 OR ID=835353 OR ID=750408 OR ID=1269820 OR ID=638713 OR ID=541373 OR ID=810905 OR ID=1173628 OR ID=1176423 OR ID=1350261 OR ID=1479089 OR ID=730848 OR ID=1387726 OR ID=745765 OR ID=778397 OR ID=1133205 OR ID=642228 OR ID=1404310 OR ID=1378428 OR ID=829525 OR ID=1148880 OR ID=1253942 OR ID=1408644 OR ID=619110 OR ID=1486823 OR ID=1568019 OR ID=1083784 OR ID=1236556 OR ID=1707079 OR ID=951209 OR ID=1350243 OR ID=1641787 OR ID=555038 OR ID=1616234 OR ID=1595849 OR ID=765129 OR ID=630056 OR ID=1099837 OR ID=1447905 OR ID=1199928 OR ID=708595 OR ID=1379298 OR ID=1271261 OR ID=1103182 OR ID=495045 OR ID=1790201 OR ID=742549 OR ID=952766 OR ID=896840 OR ID=999218 OR ID=1616686 OR ID=827121 OR ID=747084 OR ID=1251464 OR ID=709949 OR ID=861184 OR ID=960382 OR ID=1119373 OR ID=1113038 OR ID=829525 OR ID=1234550 OR ID=1508979 OR ID=1595849 OR ID=765129 OR ID=630056 OR ID=1099837 OR ID=565309 OR ID=1532236 OR ID=896840 OR ID=885152 OR ID=1242130 OR ID=827121 OR ID=709949 OR ID=1168068 OR ID=1030530 OR ID=893089 OR ID=1536850 OR ID=1645316 OR ID=845243 OR ID=1268725 OR ID=654093 OR ID=1562170 OR ID=1572075 OR ID=604836 OR ID=633289 OR ID=1309138 OR ID=1458973 OR ID=601160 OR ID=1272470 OR ID=1407641 OR ID=531291 OR ID=919473 OR ID=995821 OR ID=1496840 OR ID=1287531 OR ID=826826 OR ID=985548 OR ID=1509720 OR ID=1522240 OR ID=1307149 OR ID=1308075 OR ID=751821 OR ID=1273282 OR ID=1183155 OR ID=862244 OR ID=1013228 OR ID=1091031 OR ID=1514753 OR ID=1519597 OR ID=1168200 OR ID=1050889 OR ID=818020 OR ID=1790201 OR ID=742549 OR ID=1415335 OR ID=1436035 OR ID=1251464 OR ID=1602852 OR ID=1278730 OR ID=960382 OR ID=835651 OR ID=682505 OR ID=1119373 OR ID=1649067 OR ID=1439824 OR ID=1406660 OR ID=1172820 OR ID=1687479 OR ID=1529527 OR ID=823764 OR ID=627924 OR ID=1170221 OR ID=607906 OR ID=1355667 OR ID=1452953 OR ID=1325275 OR ID=1589165 OR ID=887197 OR ID=1769480 OR ID=1238258 OR ID=1133434 OR ID=578533 OR ID=653475 OR ID=960025 OR ID=1345786 OR ID=1124670 OR ID=1649029 OR ID=861141 OR ID=1553734 OR ID=770286 OR ID=978080 OR ID=1728231 OR ID=1186028 OR ID=1234895 OR ID=640983 OR ID=816834 OR ID=494720 OR ID=752896 OR ID=1567015 OR ID=846862 OR ID=901098 OR ID=703931 OR ID=1218241 OR ID=1698916 OR ID=810931 OR ID=1199490 OR ID=1790201 OR ID=742549 OR ID=1148238 OR ID=1251464 OR ID=960382 OR ID=1119373 OR ID=1649067 OR ID=993666 OR ID=1406660 OR ID=1230214 OR ID=871704 OR ID=1202062 OR ID=1529527 OR ID=1060938 OR ID=823764 OR ID=627924 OR ID=861519 OR ID=1545435 OR ID=1452953 OR ID=1325275 OR ID=887197 OR ID=1769480 OR ID=1475416 OR ID=1238258 OR ID=1133434 OR ID=578533 OR ID=1642106 OR ID=1345786 OR ID=1124670 OR ID=1649029 OR ID=887969 OR ID=861141 OR ID=1050889 OR ID=595809 OR ID=770516 OR ID=1728231 OR ID=1186028 OR ID=1766587 OR ID=1778994 OR ID=690620 OR ID=640983 OR ID=966195 OR ID=829525 OR ID=1234550 OR ID=1771987 OR ID=1595849 OR ID=765129 OR ID=630056 OR ID=1099837 OR ID=1532236 OR ID=690121 OR ID=1113029 OR ID=896840 OR ID=885152 OR ID=827121 OR ID=709949 OR ID=671250 OR ID=842467 OR ID=1168068 OR ID=1030530 OR ID=893089 OR ID=1536850 OR ID=1184459 OR ID=845243 OR ID=1268725 OR ID=629740 OR ID=1735646 OR ID=654093 OR ID=893974 OR ID=1572075 OR ID=604836 OR ID=633289 OR ID=1309138 OR ID=1458973 OR ID=1547049 OR ID=1272470 OR ID=1407641 OR ID=1279469 OR ID=995821 OR ID=1496840 OR ID=1287531 OR ID=985548 OR ID=1509720 OR ID=1522240 OR ID=1307149 OR ID=1526845 OR ID=1278746 OR ID=676734 OR ID=1183155 OR ID=470780 OR ID=1541018 OR ID=862244 OR ID=1547201 OR ID=1547196 OR ID=1724304 OR ID=1724137 OR ID=1516174 OR ID=1547210 OR ID=1285655 OR ID=1529201 OR ID=1289174 OR ID=1289099 OR ID=1547224 OR ID=562770 OR ID=829525 OR ID=1289135 OR ID=1736340 OR ID=1737786 OR ID=1288985 OR ID=1751048 OR ID=1737763 OR ID=1288904 OR ID=1697816 OR ID=1192007 OR ID=654010 OR ID=1543722 OR ID=1289139 OR ID=1723877 OR ID=608284 OR ID=674477 OR ID=712325 OR ID=450437 OR ID=716481 OR ID=1289152 OR ID=1530487 OR ID=1508979 OR ID=578567 OR ID=1407076 OR ID=1529267 OR ID=673190 OR ID=956398 OR ID=1705857 OR ID=502392 OR ID=1693520 OR ID=1595849 OR ID=765129 OR ID=1247741 OR ID=630056 OR ID=1249077 OR ID=1099837 OR ID=608254 OR ID=928484 OR ID=598370 OR ID=691795 OR ID=679049 OR ID=1017046 OR ID=589358 OR ID=950412 OR ID=1206005 OR ID=849900 OR ID=1035895 OR ID=908521 OR ID=1771806 OR ID=1084009 OR ID=858735 OR ID=939890 OR ID=1103573 OR ID=1742904 OR ID=1105382 OR ID=1308757 OR ID=959620 OR ID=1135382 OR ID=1087279 OR ID=1204732 OR ID=1487047 OR ID=969257 OR ID=1476363 OR ID=849976 OR ID=768748 OR ID=761029 OR ID=881281 OR ID=1603252 OR ID=1220235 OR ID=1662827 OR ID=1610399 OR ID=880514 OR ID=1281369 OR ID=1035992 OR ID=858758 OR ID=859119 OR ID=975126 OR ID=1035911 OR ID=975680 OR ID=990068 OR ID=1335511 OR ID=945719 OR ID=908994 OR ID=1683836 OR ID=962560 OR ID=849981 OR ID=1384396 OR ID=878578 OR ID=967819 OR ID=791750 OR ID=1059895 OR ID=881871 OR ID=789811 OR ID=767778 OR ID=935224 OR ID=876985 OR ID=828294 OR ID=817297 OR ID=1499116 OR ID=1672478 OR ID=739283 OR ID=1531795 OR ID=695347 OR ID=844088 OR ID=1688740 OR ID=588990 OR ID=828207 OR ID=1137178 OR ID=648708 OR ID=583890 OR ID=1353881 OR ID=1616959 OR ID=552026 OR ID=758510 OR ID=1164685 OR ID=660437 OR ID=828217 OR ID=906868 OR ID=567801 OR ID=813576 OR ID=833290 OR ID=1639668 OR ID=561128 OR ID=995562 OR ID=1139605 OR ID=1244366 OR ID=1348269 OR ID=1635192 OR ID=744801 OR ID=786491 OR ID=1017642 OR ID=1384045 OR ID=1727904 OR ID=1724728 OR ID=816150 OR ID=1170686 OR ID=823624 OR ID=502893 OR ID=796224 OR ID=1358525 OR ID=682310 OR ID=508002 OR ID=924898 OR ID=1358531 OR ID=1522476 OR ID=912207 OR ID=1523515 OR ID=1523147 OR ID=706964 OR ID=1358530 OR ID=589554 OR ID=1563068 OR ID=1355015 OR ID=1405985 OR ID=1210539 OR ID=1358527 OR ID=1637385 OR ID=1013063 OR ID=1075626 OR ID=588949 OR ID=670204 OR ID=1264618 OR ID=1736424 OR ID=924334 OR ID=1674535 OR ID=1320736 OR ID=946558 OR ID=1662766 OR ID=1358545 OR ID=847975 OR ID=1358535 OR ID=1527672 OR ID=1475905 OR ID=1278684 OR ID=1132147 OR ID=1281065 OR ID=1542350 OR ID=1523838 OR ID=1365870 OR ID=1779243 OR ID=1405972 OR ID=1107851 OR ID=901851 OR ID=1776086 OR ID=753637 OR ID=1073098 OR ID=1538884 OR ID=1787093 OR ID=829525 OR ID=579255 OR ID=1595849 OR ID=765129 OR ID=630056 OR ID=1099837 OR ID=896840 OR ID=827121 OR ID=709949 OR ID=651571 OR ID=618916 OR ID=1168068 OR ID=893089 OR ID=1536850 OR ID=845243 OR ID=1268725 OR ID=1234550 OR ID=687609 OR ID=654093 OR ID=1572075 OR ID=604836 OR ID=633289 OR ID=1309138 OR ID=1458973 OR ID=1272470 OR ID=1407641 OR ID=995821 OR ID=1496840 OR ID=1287531 OR ID=626737 OR ID=985548 OR ID=1509720 OR ID=1522240 OR ID=1532236 OR ID=760637 OR ID=1183155 OR ID=862244 OR ID=1013228 OR ID=621492 OR ID=599837 OR ID=803621 OR ID=1097253 OR ID=1775401 OR ID=1303686 OR ID=885152 OR ID=1258996 OR ID=1753443 OR ID=633565 OR ID=1458785 OR ID=1469438"""
            conn.execute(select_query, commit=False)
            while True:
                posts = conn.fetchmany(size=1)
                if not posts:
                    logger.info("No more posts to parse.")
                    break

                for post in posts:
                    post_id, title, body = post
                    text_blocks = parser.parse(body)
                    if text_blocks is None:
                        continue

                    print(f"Post ID: {post_id}")
                    print(f"Body:\n {text_blocks}")

                    title = title + " " if title is not None else ""
                    print(
                        title
                        + " ".join([x.text for x in text_blocks if x.text is not None])
                    )
                    print("\n")
                    for x in text_blocks:
                        if x.text is None:
                            print("ERR", post_id, "title", title, "body", body, "x", x)

                    with open(".cache/test_documents.txt", "a") as f:
                        f.write(
                            f"{post_id}\t{title}\t{' '.join([x.text for x in text_blocks if x.text is not None])}\n"
                        )

                # size=1should_continue = input("Continue? [(y)/n]: ")
                should_continue = "y"
                if should_continue.lower() == "n":
                    break
    else:
        test_htmls = [
            """
        <html>
        <h2>Test the handling fo <strong>strong or <i>bold and italic</i></strong> text</h2>
        <p>This is a piece of text with <strong>typing</strong> but also tailing text. And then some extra <strong>text</strong> for laughs.</p>
        </html>
        """,
            """<html>\n  <body><p>You should implement <a href="https://api.drupal.org/api/drupal/modules%21node%21node.api.php/function/hook_node_presave/7" rel="nofollow"><code>hook_node_presave</code></a> to set the values you need to change there.</p>\n\n<p>Code sample:</p>\n\n<pre><code>function MODULE_node_presave($node) {\n    if($node-&gt;type === \'MY_NODE_TYPE\') \n        $node-&gt;uid = 1;\n}\n</code></pre>\n</body>\n</html>\n""",
        ]

        for test_html in test_htmls:
            text_blocks = parser.parse(test_html)
            pprint.pp(text_blocks)
            print(test_html)
