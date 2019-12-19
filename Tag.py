class Tag:

    # these are the nodes used in the compositional parser which store part of speech, sentiment score, and backpointer
    def __init__(self, p_o_s, sent, backpointers):
        self.pos = p_o_s
        self.sentiment = sent
        self.bp = backpointers

    def get_pos(self):
        return self.pos

    def get_sentiment(self):
        return self.sentiment

    def get_bp(self):
        return self.bp
