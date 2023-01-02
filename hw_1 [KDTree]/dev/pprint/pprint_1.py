import pprint

A = {}
B = {"link": A}
A["link"] = B
print(A)
pprint.pprint(A)


#
# def truncateFloat(data):
#     return tuple(
#         [x if isinstance(x, float) else (x if not isinstance(x, tuple) else truncateFloat(x)) for x in data])
#
#
# pprint(truncateFloat(content))
