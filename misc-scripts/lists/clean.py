import os
import re 
my_file = open("multi-class-list.txt","r")
class_list = []
for item in my_file:
    item = str(item)
    m = re.search('/(.+?)/',item)
    if m:
        found = m.group(1)
    class_list.append(found)
print(class_list)
f = open("multi-class.txt","w")
for class_ in class_list:
    f.write(str(class_))
    f.write("\n")
