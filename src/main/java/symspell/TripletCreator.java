package symspell;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TripletCreator {
    String line;
    int length, maxEdit;
    long count;
    List<SuggestItem> items;
    List<Triplet> triplets = new ArrayList<>();

    public TripletCreator() throws IOException {
        SymSpell symSpell = new SymSpell(-1, 3,-1, 10);
        if(!symSpell.loadDictionary("data/wiki_freq_dict.txt", 0, 1))
            throw new FileNotFoundException();
        BufferedReader br = new BufferedReader(new FileReader("data/wiki_freq_dict.txt"));

        while((line = br.readLine()) != null){
            if(line.isEmpty()) continue;
            line = line.split(" ")[0];
            length = line.length();
            if(length > 12) maxEdit = 3;
            else if(length > 4) maxEdit = 2;
            else maxEdit = 1;
            items = symSpell.lookup(line, SymSpell.Verbosity.All);
            if(items.isEmpty()) continue;
            Collections.sort(items);
            count = items.get(0).count / 10;
//            items.remove(0);
            items.removeIf(it -> (it.distance > maxEdit || it.count > count));  // intended word has to be 10x more likely
            for(SuggestItem item : items) triplets.add(new Triplet(line, item.term, (int)item.count));
        }

        BufferedWriter bw = new BufferedWriter(new FileWriter("data/triplets.csv"));
        for(Triplet tp : triplets) bw.write(tp.toString());
        bw.close();
    }

    public static void main(String[] args) throws IOException {
        TripletCreator trp = new TripletCreator();
    }

    public class Triplet{
        public String intended, observed;
        int count;

        public Triplet(String intended, String observed, int count){
            this.intended = intended;
            this.observed = observed;
            this.count = count;
        }

        @Override
        public String toString() {
            return String.format("(%s,,,%s,,,%d)", intended, observed, count);
        }
    }
}
