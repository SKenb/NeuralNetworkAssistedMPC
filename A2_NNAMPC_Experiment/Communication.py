from opcua import Client
from opcua import ua

OPCUA_CLIENT1 = None
def connectToOPCUAServer1(url = "opc.tcp://143.50.141.214:4840"):
    global OPCUA_CLIENT1
    if OPCUA_CLIENT1: 
        print("Already connected")
        return

    OPCUA_CLIENT1 = Client(url)
    OPCUA_CLIENT1.connect()
    print("Connected to OPC UA Server")

def readFromUPCUAServer1():
    global OPCUA_CLIENT1
    if not OPCUA_CLIENT1:
        print("OPC-UA Cleint1 not yet connected")
        return None
    

    node = OPCUA_CLIENT1.get_node("ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.HPLC_5_SETP.Value") 
    value = node.get_value()
    print(f"Value: {value}")


OPCUA_CLIENT2 = None
def connectToOPCUAServer2(url = "opc.tcp://143.50.141.63:36090"):
    global OPCUA_CLIENT2
    if OPCUA_CLIENT2: 
        print("Already connected")
        return

    OPCUA_CLIENT2 = Client(url)
    OPCUA_CLIENT2.connect()
    print("Connected to OPC UA Server")

OPCUA_CLIENT3 = None
def connectToOPCUAServer3(url = "opc.tcp://localhost:4840"):
    global OPCUA_CLIENT3
    if OPCUA_CLIENT3: 
        print("Already connected")
        return

    OPCUA_CLIENT3 = Client(url)
    OPCUA_CLIENT3.connect()
    print("Connected to OPC UA Server")

def writeToUPCUAServer2():
    global OPCUA_CLIENT2

    if not OPCUA_CLIENT2:
        print("OPC-UA Client2 not connected")
        return
    
    try:
        node = OPCUA_CLIENT2.get_node("ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.HPLC_5_SETP.Value") 
        node.set_value(1.0)
        print(f"Successfully wrote values")

    except Exception as e:
        print(f"Failed writting: {str(e)}")


def readAllRequiredDataFromUPCUAServers():
    global OPCUA_CLIENT1, OPCUA_CLIENT2, OPCUA_CLIENT3
 
    
    return {
        "inputs": {
            "C3meas": max(0, getValueByBrowseName(OPCUA_CLIENT2, "Link 2|product1")), # getValueOfNode(OPCUA_CLIENT2, "ns=2;g=9cec426b-0293-8231-9f28-5e90740768e2"),
            "C3ref": float(getValueByBrowseName(OPCUA_CLIENT3, "C3ref")),
            "temperature": 140,
            "flowRate": 1
        }
    }

#ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.HPLC_5_SETP.Value

def getValueOfNode(client, nodePath):
    try:
        #node = client.root.get_child(nodePath)
        node = client.get_node(nodePath) 
        return node.get_value()
    
    except Exception as e:
        print(f"Failed reading: {str(e)}")
        return None
    
def getValueByBrowseName(client, browse_name_to_find):
    if not client:
        print("Error")
        return None
    
    root_node = client.get_objects_node()
    namespace_index = None  # Set to an int like 2 if you want to restrict to one namespace

    node = find_node_by_browse_name_recursive(root_node, browse_name_to_find, namespace_index)
        
    if not node:
        print("Node not found")
        return None
    
    return node.get_value()
    

def writeAllValues(C1, C2, flowRate, temperature):
    global OPCUA_CLIENT1, OPCUA_CLIENT2

    def writeValue(client, node, value):
        if not client:
            print("OPC-UA Client2 not connected")
            return
        
        try:
            node = client.get_node(node) 
            node.set_value(value)
            #print(f"Successfully wrote values")

        except Exception as e:
            print(f"Failed writting: {str(e)}")


    # Calculate flowRates
    C1_STOCK = 1.0
    C2_STOCK = 1.0

    pumpC1_value = C1 / C1_STOCK * flowRate
    pumpC2_value = C2 / C2_STOCK * flowRate
    pumpC0_value = flowRate - pumpC1_value - pumpC2_value

    if pumpC0_value < 0:
        return "[ERR]: Illegal flow rates - Data not sent to OPC-UA"

    writeValue(OPCUA_CLIENT1, "ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.HPLC_5_SETP.Value", float(pumpC0_value))
    writeValue(OPCUA_CLIENT1, "ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.HPLC_2_SETP.Value", float(pumpC1_value))
    writeValue(OPCUA_CLIENT1, "ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.HPLC_1_SETP.Value", float(pumpC2_value))

    writeValue(OPCUA_CLIENT1, "ns=1;s=SYNTHESISCONTROL_3:OPC_UA_SERVER.THERMO_3_SETP.Value", float(temperature))

    return f"Wrote values P1-C0: {pumpC0_value}, P2-C1: {pumpC1_value}, P3-C2: {pumpC2_value}, Temp: {temperature}"


def find_node_by_browse_name_recursive(node, target_browse_name, namespace_index=None):
    """
    Recursively search for a node with a given BrowseName.
    Optionally filter by namespace index.
    """
    try:
        for child in node.get_children():
            browse_name = child.get_browse_name()
            if "Server" in browse_name.Name: continue

            if browse_name.Name == target_browse_name:
                if namespace_index is None or browse_name.NamespaceIndex == namespace_index:
                    return child  # Found the node

            # Recurse into child nodes
            result = find_node_by_browse_name_recursive(child, target_browse_name, namespace_index)
            if result:
                return result

    except Exception as e:
        # Catch unexpected access errors in some nodes
        print(f"⚠️ Error browsing node: {e}")

    return None  # Not found
