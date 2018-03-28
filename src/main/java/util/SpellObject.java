package util;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

public class SpellObject {
    public long osmId;
    public Optional<String> currentName;
    public Set<String> corrections;

    public SpellObject(long osmId) {
        this.osmId = osmId;
        corrections = new HashSet<>();
        currentName = Optional.empty();
    }

    public void addSuggestion(String suggestion) {
        corrections.add(suggestion);
    }

    public void addName(String name) {
        currentName = Optional.of(name);
    }

    public void print() {
        if (corrections.isEmpty())
            return;
        StringBuilder sb = new StringBuilder();
        sb.append(osmId).append(",,,");
        currentName.ifPresent(sb::append);
        corrections.forEach(c -> sb.append(",,,").append(c));
        System.out.println(sb.toString());
    }
}
