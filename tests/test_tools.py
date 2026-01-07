# test_tools.py (Aynı kalabilir, sadece importların hatasız olduğundan emin ol)
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.tools.search_tools import point_search_tool, broad_search_tool

def run_tests():
    load_dotenv()
    print("--- TEST BAŞLIYOR ---")
    
    # Test 1
    print("\n1. POINT SEARCH (Tanım)")
    print(point_search_tool.invoke("kvkk madde 11 nedir"))
    
    # Test 2
    print("\n2. BROAD SEARCH (Çeşitlilik)")
    # Burada tek sorgu ile hem Yönetmelik hem İlke Kararı gelmeli
    #print(broad_search_tool.invoke("Birden fazla belgeye göre veri sorumlusu nedir?"))

if __name__ == "__main__":
    run_tests()