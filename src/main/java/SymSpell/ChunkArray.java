package SymSpell;

import SymSpell.SymSpell.SuggestionStage.Node;

import java.util.Arrays;

// A growable list of elements that's optimized to support adds, but not deletes,
// of large numbers of elements, storing data in a way that's friendly to the garbage
// collector (not backed by a monolithic array object), and can grow without needing
// to copy the entire backing array contents from the old backing array to the new.
public class ChunkArray<T>
{
    private static int chunkSize = 4096; //this must be a power of 2, otherwise can't optimize row and col functions
    private static int divShift = 12; // number of bits to shift right to do division by chunkSize (the bit position of chunkSize)
    public Node[][] values;// { get; private set; }
    public int count;// { get; private set; }

    public ChunkArray(int initialCapacity)
    {
        int chunks = (initialCapacity + chunkSize - 1) / chunkSize;
        values = new Node[chunks][];
        for (int i = 0; i < values.length; i++) values[i] = new Node[chunkSize];
    }

    public int add(Node value)
    {
        if (count == capacity()) {
            Node[][] newValues = Arrays.copyOf(values, values.length + 1);
            newValues[values.length] = new Node[chunkSize];
            values = newValues;
        }

        values[row(count)][col(count)] = value;
        count++;
        return count - 1;
    }

    public void clear()
    {
        count = 0;
    }

    public Node getValues(int index) {
        return values[row(index)][col(index)];
    }
    public void setValues(int index, Node value){
        values[row(index)][col(index)] = value;
    }
    public void setValues(int index, Node value, Node[][] list){
        list[row(index)][col(index)] = value;
    }

    private int row(int index) { return index >> divShift; } // same as index / chunkSize
    private int col(int index) { return index & (chunkSize - 1); } //same as index % chunkSize
    private int capacity() { return values.length * chunkSize; }
}