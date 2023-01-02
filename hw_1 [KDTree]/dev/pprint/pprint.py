from pprint import pprint
from urllib import request
import json
from pprint import PrettyPrinter
from pprint import pformat

path = "https://jsonplaceholder.typicode.com/users"
response = request.urlopen(path)
json_response = response.read()
content = json.loads(json_response)
# pprint(content, depth=2, indent=2, width=40, compact=True)  # width=1 => every component on a new line

# with open('content.txt', 'w') as f:
#     pprint(content, depth=2, indent=2, width=40, compact=True, stream=f)

# Although dictionaries are generally considered unordered data structures,
# since Python 3.6, dictionaries are ordered by insertion.

pp = PrettyPrinter(depth=1, indent=4, width=40, compact=True)
# pp.pprint(content)

content = pp.pformat(content)
print(content)
