import sys
import io
import json
from pprint import pprint

def main():
    print (sys.argv)
    if len(sys.argv) > 2:
        print("Invalid Entry!")
        print("Expect arguments in format <python sieveson.py JSONFILE> ")
    else:
        filterJSON(sys.argv[1])

def filterJSON(path):
    result = []
    with open(path, 'r') as r:
        raw = json.load(r)
    for i in range(0, len(raw)):
        current = {}
        current['sequence'] = raw[i]['protein']['sequence']
        current['consensus'] = raw[i]['protein']['mobidb']['consensus']['predictors']
        result.append(current)
    print i
    with open('filterJSON.json','w') as w:
        json.dump(result, w)
    print ("Finished, created filtered json file <filterJSON.json>")


if __name__ == "__main__":
    main()
