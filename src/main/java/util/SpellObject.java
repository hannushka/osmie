package util;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class SpellObject {
    public long osmId;
    public Optional<String> currentName;
    public List<String> corrections;

    public SpellObject(long osmId) {
        this.osmId = osmId;
        corrections = new ArrayList<>();
        currentName = Optional.empty();
    }

    public void addSuggestion(String suggestion) {
        corrections.add(suggestion);
    }

    public void addName(String name) {
        currentName = Optional.of(name);
    }

    public void print() {
        StringBuilder sb = new StringBuilder();
        sb.append(osmId);
        if (currentName.isPresent())
            sb.append(",,,").append(currentName.get());
        corrections.forEach(c -> sb.append(",,,").append(c));
        System.out.println(sb.toString());
    }
}
