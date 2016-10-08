import json
d = json.dumps({ \
"proportion" :  0.003, \
"sketchNum" :  40,\
"matFileName" :  u"../data/MultiPieGRAY.mat",\
"threadhold" :  0.155,\
"trainErr" : True\
},sort_keys=True,indent=4)

print d

jf = file("arg.json")
arg = json.load(jf)
print arg
