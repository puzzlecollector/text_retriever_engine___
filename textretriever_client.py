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
        print("listing") 
        print()
        print() 
        myinput = ListingRequest()  
        myinput.begin_id = "0" 
        myinput.count = 2 
        out = self.stub.ListCandidates(myinput) 
        print(out.candidates)  

        print("adding or updating request") 
        print() 
        print() 
        myinput = AddOrUpdateRequest() 
        myinput.candidates[:] = [Candidate(text = "이건 새로운 텍스트야")] 
        self.stub.AddOrUpdateCandidates(myinput)  
        print("done adding!") 
        list_test = ListingRequest() 
        list_test.begin_id = "0" 
        list_test.count = 10 # there has to be 10 elements after addition 
        out = self.stub.ListCandidates(list_test) 
        print(out.candidates) 

        print("deleting") 
        print()
        print() 
        myinput = DeletionRequest()
        myinput.ids[:] = ["3", "5"] # get rid of id = 3, 5
        self.stub.DeleteCandidates(myinput) 
        print("done deleting!") 
        list_test = ListingRequest() 
        list_test.begin_id = "0" 
        list_test.count = 9 # there should be 9 elements after deletion 
        out = self.stub.ListCandidates(list_test) 
        print(out.candidates) 
        
        

    
if __name__ == "__main__":
    service_client = TextRetrieverClient()
    print("Output")
    print("*" * 100)
    print(service_client.processor()) ## simple test 
    print("="*30 + "Testing other functionalities" + "="*30) 
    service_client.test_other_utils() 
