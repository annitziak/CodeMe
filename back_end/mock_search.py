import random

SEARCH_RESULTS = [
    """{
        "has_next": false,
        "has_prev": false,
        "page": 0,
        "page_size": 20,
        "result": [
            {
                "answer_count": 1,
                "body": "TO BE ADDED",
                "comment_count": 0,
                "creation_date": 1414885423,
                "doc_id": 26694464,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 10.40335464477539,
                "tags": "|python|windows|subprocess|popen|",
                "title": "TO BE ADDED",
                "view_count": 66,
            },
            {
                "answer_count": 2,
                "body": "TO BE ADDED",
                "comment_count": 0,
                "creation_date": 1414566373,
                "doc_id": 26624498,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 10.183927536010742,
                "tags": "|java|python|apache-storm|",
                "title": "TO BE ADDED",
                "view_count": 1639,
            },
            {
                "answer_count": 2,
                "body": "TO BE ADDED",
                "comment_count": 1,
                "creation_date": 1414624325,
                "doc_id": 26642265,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 9.581992149353027,
                "tags": "|python|linux|terminal|argparse|pipeline|",
                "title": "TO BE ADDED",
                "view_count": 1879,
            },
            {
                "answer_count": 4,
                "body": "TO BE ADDED",
                "comment_count": 1,
                "creation_date": 1414748018,
                "doc_id": 26671368,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 9.53388500213623,
                "tags": "|python|c++|ctypes|",
                "title": "TO BE ADDED",
                "view_count": 218,
            },
            {
                "answer_count": 1,
                "body": "TO BE ADDED",
                "comment_count": 0,
                "creation_date": 1414868820,
                "doc_id": 26692052,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 9.487017631530762,
                "tags": "|python|python-3.x|pycharm|",
                "title": "TO BE ADDED",
                "view_count": 4346,
            },
            {
                "answer_count": 0,
                "body": "TO BE ADDED",
                "comment_count": 1,
                "creation_date": 1414567035,
                "doc_id": 26624654,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 9.382309913635254,
                "tags": "",
                "title": "TO BE ADDED",
                "view_count": 0,
            },
            {
                "answer_count": 0,
                "body": "TO BE ADDED",
                "comment_count": 0,
                "creation_date": 1414836305,
                "doc_id": 26688123,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 9.209115028381348,
                "tags": "",
                "title": "TO BE ADDED",
                "view_count": 0,
            },
            {
                "answer_count": 1,
                "body": "TO BE ADDED",
                "comment_count": 3,
                "creation_date": 1414754655,
                "doc_id": 26673556,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 8.921977996826172,
                "tags": "|python|linux|bash|shell|raspberry-pi|",
                "title": "TO BE ADDED",
                "view_count": 717,
            },
            {
                "answer_count": 0,
                "body": "TO BE ADDED",
                "comment_count": 0,
                "creation_date": 1414767096,
                "doc_id": 26677499,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 8.88099479675293,
                "tags": "",
                "title": "TO BE ADDED",
                "view_count": 0,
            },
            {
                "answer_count": 1,
                "body": "TO BE ADDED",
                "comment_count": 2,
                "creation_date": 1414784310,
                "doc_id": 26682214,
                "favorite_count": 0,
                "ownerdisplayname": "",
                "score": 8.815267562866211,
                "tags": "|python|c|module|extending|",
                "title": "TO BE ADDED",
                "view_count": 515,
            },
        ],
        "total_results": 10,
    }
    """,
    """
    {"has_next":false,"has_prev":false,"page":0,"page_size":20,"result":[{"answer_count":2,"body":"TO BE ADDED","comment_count":4,"creation_date":1414760148,"doc_id":26675250,"favorite_count":0,"ownerdisplayname":"","score":13.43576717376709,"tags":"|ruby-on-rails|","title":"TO BE ADDED","view_count":161},{"answer_count":2,"body":"TO BE ADDED","comment_count":0,"creation_date":1414683643,"doc_id":26657360,"favorite_count":0,"ownerdisplayname":"","score":11.657254219055176,"tags":"|ios|in-app-purchase|icloud|","title":"TO BE ADDED","view_count":81},{"answer_count":0,"body":"TO BE ADDED","comment_count":1,"creation_date":1414774861,"doc_id":26679852,"favorite_count":0,"ownerdisplayname":"","score":10.934154510498047,"tags":"|android|deployment|in-app-purchase|updates|subscription|","title":"TO BE ADDED","view_count":72},{"answer_count":0,"body":"TO BE ADDED","comment_count":3,"creation_date":1414637454,"doc_id":26644244,"favorite_count":0,"ownerdisplayname":"","score":10.783902168273926,"tags":"","title":"TO BE ADDED","view_count":0},{"answer_count":1,"body":"TO BE ADDED","comment_count":0,"creation_date":1414777694,"doc_id":26680604,"favorite_count":0,"ownerdisplayname":"","score":10.42983341217041,"tags":"|java|hibernate|jpa|predicate|specifications|","title":"TO BE ADDED","view_count":1555},{"answer_count":1,"body":"TO BE ADDED","comment_count":1,"creation_date":1414634252,"doc_id":26643796,"favorite_count":0,"ownerdisplayname":"","score":10.145895957946777,"tags":"|python|","title":"TO BE ADDED","view_count":316},{"answer_count":1,"body":"TO BE ADDED","comment_count":5,"creation_date":1414674326,"doc_id":26653771,"favorite_count":0,"ownerdisplayname":"","score":9.696788787841797,"tags":"|javascript|android|knockout.js|","title":"TO BE ADDED","view_count":188},{"answer_count":0,"body":"TO BE ADDED","comment_count":5,"creation_date":1414747760,"doc_id":26671283,"favorite_count":0,"ownerdisplayname":"","score":9.58792495727539,"tags":"","title":"TO BE ADDED","view_count":0},{"answer_count":0,"body":"TO BE ADDED","comment_count":0,"creation_date":1414693641,"doc_id":26660653,"favorite_count":0,"ownerdisplayname":"","score":9.518962860107422,"tags":"","title":"TO BE ADDED","view_count":0},{"answer_count":1,"body":"TO BE ADDED","comment_count":5,"creation_date":1414747976,"doc_id":26671354,"favorite_count":0,"ownerdisplayname":"","score":8.913065910339355,"tags":"|python|arguments|pyside|","title":"TO BE ADDED","view_count":4116}],"total_results":10}
    """,
    """
    {"has_next":false,"has_prev":false,"page":0,"page_size":20,"result":[{"answer_count":0,"body":"TO BE ADDED","comment_count":5,"creation_date":1414672087,"doc_id":26652990,"favorite_count":0,"ownerdisplayname":"","score":17.787370681762695,"tags":"|java|rest|","title":"TO BE ADDED","view_count":535},{"answer_count":3,"body":"TO BE ADDED","comment_count":3,"creation_date":1414566507,"doc_id":26624528,"favorite_count":0,"ownerdisplayname":"","score":16.157957077026367,"tags":"|android|","title":"TO BE ADDED","view_count":85},{"answer_count":2,"body":"TO BE ADDED","comment_count":10,"creation_date":1414883713,"doc_id":26694264,"favorite_count":0,"ownerdisplayname":"","score":15.605340003967285,"tags":"|java|exception|nullpointerexception|lang|","title":"TO BE ADDED","view_count":1980},{"answer_count":2,"body":"TO BE ADDED","comment_count":5,"creation_date":1414777801,"doc_id":26680637,"favorite_count":0,"ownerdisplayname":"","score":15.217535018920898,"tags":"|android|intellij-idea|nullpointerexception|libgdx|game-engine|","title":"TO BE ADDED","view_count":1864},{"answer_count":0,"body":"TO BE ADDED","comment_count":2,"creation_date":1414884868,"doc_id":26694396,"favorite_count":0,"ownerdisplayname":"","score":15.147391319274902,"tags":"","title":"TO BE ADDED","view_count":0},{"answer_count":0,"body":"TO BE ADDED","comment_count":5,"creation_date":1414788405,"doc_id":26683154,"favorite_count":0,"ownerdisplayname":"","score":14.954310417175293,"tags":"","title":"TO BE ADDED","view_count":0},{"answer_count":0,"body":"TO BE ADDED","comment_count":0,"creation_date":1414711417,"doc_id":26665139,"favorite_count":0,"ownerdisplayname":"","score":13.464801788330078,"tags":"","title":"TO BE ADDED","view_count":0},{"answer_count":1,"body":"TO BE ADDED","comment_count":19,"creation_date":1414519673,"doc_id":26615426,"favorite_count":0,"ownerdisplayname":"","score":13.422738075256348,"tags":"|java|mysql|jsp|","title":"TO BE ADDED","view_count":9595},{"answer_count":1,"body":"TO BE ADDED","comment_count":0,"creation_date":1414566216,"doc_id":26624455,"favorite_count":0,"ownerdisplayname":"","score":13.280847549438477,"tags":"|java|import|","title":"TO BE ADDED","view_count":223},{"answer_count":1,"body":"TO BE ADDED","comment_count":1,"creation_date":1414930837,"doc_id":26699227,"favorite_count":0,"ownerdisplayname":"","score":13.131599426269531,"tags":"|java|json|jackson|","title":"TO BE ADDED","view_count":7540}],"total_results":10}
    """,
]

QUERIES = [
    "TypeError: 'NoneType' object is not subscriptable in Python",
    "how good is python as a programming langauge",
    "NullPointerException handling in Java",
]


class MockSearch:
    def __init__(self):
        self.search_results = SEARCH_RESULTS

    def search(self, query):
        return random.choice(self.search_results)

    def advanced_search(self, query):
        return random.choice(self.search_results)
