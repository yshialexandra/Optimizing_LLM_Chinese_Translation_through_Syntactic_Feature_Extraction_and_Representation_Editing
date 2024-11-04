import stanza
import xml.etree.ElementTree as ET
from nltk.tree import Tree


class SentenceParser:
    def __init__(self, sentence):
        self.sentence_constituency = self.get_constituency_parsing(sentence)
        self.tree = Tree.fromstring(self.sentence_constituency)
        self.vector = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.nlp_dependency = stanza.Pipeline('zh')
    
    def clear(self):
        self.vector = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    
    def get_constituency_parsing(self, sentence):
        doc = self.nlp_dependency(sentence)
        sentence_constituency = str(doc.sentences[0].constituency)
        return sentence_constituency
    
    # 3.0 NP→PN
    def NP_PN(self, t=None):
        t = t if t is not None else self.tree
        if t.label() == 'NP' and len(t) >= 1:
            if t[0].label() == 'PN':
                self.vector[0] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_PN(subtree)

    # 4.0 NP→DP NP
    def NP_DP_NP(self, t=None):
        t = t if t is not None else self.tree
        if t.label() == 'NP' and len(t) >= 2:
            if t[0].label() == 'DP' and t[1].label() == 'NP':
                self.vector[1] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_DP_NP(subtree)

    # 6.2 DP→DT
    def DP_DT(self, t=None):
        t = t if t is not None else self.tree
        if t.label() == 'DP' and len(t) >= 1:
            if t[0].label() == 'DT':
                self.vector[2] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.DP_DT(subtree)

    #6.6 IP→NP VP PU
    def IP_NP_VP_PU(self, t,vector):
        if t.label() == 'IP' and len(t) >= 3:
            if t[0].label() == 'NP' and t[1].label() == 'VP' and t[2].label() == 'PU':
                vector[3] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.IP_NP_VP_PU(subtree,vector)

    #6.8 PRN→PU NP PU
    def PRN_PU_NP_PU(self, t,vector):
        if t.label() == 'PRN' and len(t) >= 3:
            if t[0].label() == 'PU' and t[1].label() == 'NP' and t[2].label() == 'PU':
                vector[4] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.PRN_PU_NP_PU(subtree,vector)

    #6.8 NP→NR
    def NP_NR(self, t,vector):
        if t.label() == 'NP' and len(t) >= 1:
            if t[0].label() == 'NR':
                vector[5] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_NR(subtree,vector)

    #10.0 CP_ADVP_IP
    def CP_ADVP_IP(self, t,vector):
        if t.label() == 'CP' and len(t) >= 2:
            if t[0].label() == 'ADVP' and t[1].label() == 'IP':
                vector[6] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.CP_ADVP_IP(subtree,vector)

    #10.6 NP_DNP_NP
    def NP_DNP_NP(self, t,vector):
        if t.label() == 'NP' and len(t) >= 2:
            if t[0].label() == 'DNP' and t[1].label() == 'NP':
                vector[7] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_DNP_NP(subtree,vector)
                
    #16.4 ADVP_CS
    def ADVP_CS(self, t,vector):
        if t.label() == 'ADVP' and len(t) >= 1:
            if t[0].label() == 'CS':
                vector[8] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.ADVP_CS(subtree,vector)

    #16.8 DNP_NP_DEG
    def DNP_NP_DEG(self, t,vector):
        if t.label() == 'DNP' and len(t) >= 2:
            if t[0].label() == 'NP' and t[1].label() == 'DEG': 
                vector[9] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.DNP_NP_DEG(subtree,vector)

    def NP_QP_DNP_NP(self, t,vector):
        if t.label() == 'NP' and len(t) >= 3:
            if t[0].label() == 'QP' and t[1].label() == 'DNP' and t[2].label() == 'NP':
                vector[10] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_QP_DNP_NP(subtree,vector)

    def NP_NP_PRN(self, t,vector):
        if t.label() == 'NP' and len(t) >= 2:
            if t[0].label() == 'NP' and t[1].label() == 'PRN':
                vector[11] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_NP_PRN(subtree,vector)

    def NP_NR_CC_NR(self, t,vector):
        if t.label() == 'NP' and len(t) >= 3:
            if t[0].label() == 'NR' and t[1].label() == 'CC' and t[2].label() == 'NR':
                vector[12] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_NR_CC_NR(subtree,vector)

    def NP_NP_CC_NP(self, t,vector):
        global score, count
        if t.label() == 'NP' and len(t) >= 3:
            if t[0].label() == 'NP' and t[1].label() == 'CC' and t[2].label() == 'NP':
                vector[13] = 1
        for subtree in t:
            if isinstance(subtree, Tree):
                self.NP_NP_CC_NP(subtree,vector)
                    
    def write_constituency_vector(self,sentence):
        
        sentence_constituency = self.get_constituency_parsing(sentence)
        #CFGR features
        tree = Tree.fromstring(sentence_constituency)
        find_NP_PN = self.NP_PN(tree,self.vector)
        find_NP_DP_NP = self.NP_DP_NP(tree,self.vector)
        find_DP_DT = self.DP_DT(tree,self.vector)
        find_IP_NP_VP_PU = self.IP_NP_VP_PU(tree,self.vector)
        find_PRN_PU_NP_PU = self.PRN_PU_NP_PU(tree,self.vector)
        find_NP_NR = self.NP_NR(tree,self.vector)
        find_CP_ADVP_IP = self.CP_ADVP_IP(tree,self.vector)
        find_NP_DNP_NP = self.NP_DNP_NP(tree,self.vector)
        find_ADVP_CS = self.ADVP_CS(tree,self.vector)
        find_DNP_NP_DEG = self.DNP_NP_DEG(tree,self.vector)
        #NP features
        find_NP_QP_DNP_NP = self.NP_QP_DNP_NP(tree,self.vector)
        find_NP_NP_PRN = self.NP_NP_PRN(tree,self.vector)
        find_NP_NR_CC_NR = self.NP_NR_CC_NR(tree,self.vector)
        find_NP_NP_CC_NP = self.NP_NP_CC_NP(tree,self.vector)

        return self.vector


    def get_dependency_parsing(self, sentences):
        parsed_words = []
        for sentence in sentences:
            sentence = sentence.strip()  # 去除多余的空白字符
            if not sentence:
                continue
            doc_dependency = self.nlp_dependency(sentence)
            parsed_words.append(doc_dependency)

    def depTripleFuncLex(self, sentence, head, relation, dependent):    # VV NSUBJ 我
        head_list = []
        # print(sentence)
        for item in sentence:
            if item.xpos == head or item.text == head and item.xpos == 'PN':
                head_list.append(item.id)

        for index in head_list:
            for item in sentence:
                if item.head == index and item.text == dependent and item.deprel == relation:
                    return True
        
        return False
    
    def write_dependency_vector(self, sentence):
        doc_dependency = self.nlp_dependency(sentence)
        a = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','nsubj','我')
        b = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','advmod','将')
        c = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','nsubj','他')
        d = self.depTripleFuncLex(doc_dependency.sentences[0].words,'NN','det','该')
        e = self.depTripleFuncLex(doc_dependency.sentences[0].words,'NR','case','的')
        f = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','nsubj','他们')
        g = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','nsubj','她')
        h = self.depTripleFuncLex(doc_dependency.sentences[0].words,'他','case','的')
        i = self.depTripleFuncLex(doc_dependency.sentences[0].words,'NN','nmod:assmod','他')
        j = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','punct','。')
        k = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','advmod','但是')
        l = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','nsubj','你')
        m = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','advmod','如果')
        n = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','mark','的')
        o = self.depTripleFuncLex(doc_dependency.sentences[0].words,'NN','det','任何')
        p = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','case','因为')
        q = self.depTripleFuncLex(doc_dependency.sentences[0].words,'NR','cc','和')
        r = self.depTripleFuncLex(doc_dependency.sentences[0].words,'NN','det','那些')
        s = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','nsubj','它')
        t = self.depTripleFuncLex(doc_dependency.sentences[0].words,'VV','dobj','它')
        if a:
            self.vector[14] = 1
        if b:
            self.vector[15] = 1
        if c:
            self.vector[16] = 1
        if d:
            self.vector[17] = 1
        if e:
            self.vector[18] = 1
        if f:
            self.vector[19] = 1
        if g:
            self.vector[20] = 1
        if h:
            self.vector[21] = 1
        if i:
            self.vector[22] = 1
        if j:
            self.vector[23] = 1
        if k:
            self.vector[24] = 1
        if l:
            self.vector[25] = 1
        if m:
            self.vector[26] = 1
        if n:
            self.vector[27] = 1
        if o:
            self.vector[28] = 1
        if p:
            self.vector[29] = 1
        if q:   
            self.vector[30] = 1
        if r:
            self.vector[31] = 1
        if s:
            self.vector[32] = 1
        if t:
            self.vector[33] = 1
        return self.vector



