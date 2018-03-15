package util;

import java.util.ArrayList;
import java.util.List;

public class SpellObject {
    public String name;
    public List<String> corrections;

    public SpellObject(String name) {
        this.name = name;
        corrections = new ArrayList<>();
    }

    public void addCorrection(String correctedName) {
        corrections.add(correctedName);
    }

    public void print() {
        StringBuilder sb = new StringBuilder();
        sb.append(name);
        for (int i = 0 ; i < corrections.size() ; i++) {
            sb.append(",,,");
            sb.append(corrections.get(i));
            if (i < corrections.size()-1)
                sb.append(",,,");
        }
        System.out.println(sb.toString());
    }
}
