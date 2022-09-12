# Object-Like Access for XML Files


Given XML:
```xml
<?xml version="1.0"?>
<root>
    <child name="child1"/>
</root>
```

Python access

```python
import untangle

doc = untangle.parse('path/to/xml.xml')

# gives hierarchical access
child_name = doc.root.child['name'] # 'child1'

```