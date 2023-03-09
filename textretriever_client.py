import grpc
import argparse
from textretriever_pb2 import *
from textretriever_pb2_grpc import TextRetrieverStub

class TextRetrieverClient:
    def __init__(self, ip="localhost", port=35015):
        self.server_ip = ip
        self.server_port = port
        self.stub = TextRetrieverStub(
            grpc.insecure_channel(self.server_ip + ":" + str(self.server_port))
        )
    def processor(self, input=None):
        myinput = TextRetrievalRequest()
        myinput.query_texts[:] = ["한 남자가 파스타를 먹는다."]
        myinput.num_candidates = 3
        out = self.stub.RetrieveTexts(myinput)
        return out.results 
    def test_other_utils(self, input=None): 
        ### list ###  
        myinput = ListingRequest()  
        myinput.begin_id = "0" 
        myinput.count = 2 
        out = self.stub.ListCandidates(myinput) 
        print(out.candidates) 

if __name__ == "__main__":
    service_client = TextRetrieverClient()
    print("Output")
    print("*" * 100)
    print(service_client.processor()) ## simple test 
    print("Testing other utilities") 
    service_client.test_other_utils()
