from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast


app = Flask(__name__)
api = Api(app)


class XRay(Resource):
    def get(self):
        data = "1"
        return {'output': data}, 200

    pass

api.add_resource(XRay, '/xray')

if __name__ == '__main__':
    app.run()  # run our Flask app