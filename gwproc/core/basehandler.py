class BaseHandler():

    def preprocess(self, data, *args, **kwargs):
        raise 'not implement'

    def inference(self, sess, data, *args, **kwargs):
        raise 'not implement'

    def postprocess(self, data, *args, **kwargs):
        raise 'not implement'
