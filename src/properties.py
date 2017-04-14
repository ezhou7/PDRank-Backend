class Properties:
    def __init__(self, parse: bool=True, annotate: bool=True, cluster: bool=True, indir_path=None):
        self.parse = parse
        self.annotate = annotate
        self.cluster = cluster
        self.indir_path = indir_path

    def set_parse(self):
        self.parse = True

    def set_annotate(self):
        self.annotate = True

    def set_cluster(self):
        self.cluster = True

    def set_path(self, indir_path):
        self.indir_path = indir_path

    def unset_parse(self):
        self.parse = False

    def unset_annotate(self):
        self.annotate = False

    def unset_cluster(self):
        self.cluster = False
