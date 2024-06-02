import sys

order = [
    "tensor_dense3bias0 1\n",
    "tensor_dense3kernel0 12\n",
    "tensor_dense2bias0 12\n",
    "tensor_dense2kernel0 144\n",
    "tensor_dense1bias0 12\n",
    "tensor_dense1kernel0 144\n",
    "tensor_densebias0 12\n",
    "tensor_densekernel0 72\n"
]


def sort_tensors_with_values(lines):

    tensors = {}
    #print(lines)
    for line in lines:
        if line.startswith('tensor'):
            current_label = line
            tensors[current_label] = 0
        elif current_label:
            tensors[current_label] = line
    sorted_lines = []
    for i in range(len(order)):
        sorted_lines.append(order[i])
        if i == len(order) - 1:
            sorted_lines.append(tensors[order[i]])
        else:
            sorted_lines.append(tensors[order[i]])
    #print(sorted_lines)
    with open(sys.argv[2], 'w') as file:
        file.writelines(sorted_lines)
    return sorted_lines
# Example usage:

if __name__ == "__main__":
   
    
    with open(sys.argv[1], 'r') as file:
        lines = file.readlines()
    result = sort_tensors_with_values(lines)
    print(result)
