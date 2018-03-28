package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.tags.names.NameTag;
import util.SpellObject;

import java.util.Set;

public class SmoothingCases {

    static void inBetweenNamedEdgesCheck(Set<EdgeContainer> filteredInEdges, Set<EdgeContainer> filteredOutEdges,
                                                 SpellObject so) {
        if (filteredInEdges.size() == 1 && filteredOutEdges.size() == 1) {
            String name1 = filteredInEdges.iterator().next().getName().trim().toLowerCase();
            String name2 = filteredOutEdges.iterator().next().getName().trim().toLowerCase();
            if (!name1.isEmpty() && !name2.isEmpty() && name1.equals(name2)) {
                so.addSuggestion(name1);
            }
        }
    }

    static void tailCaseCheck(Set<EdgeContainer> filteredEdges, SpellObject so) {
        if (filteredEdges.size() == 1) {
            Edge edge = filteredEdges.iterator().next().edge;
            if (edge.getTag(NameTag.KEY).isPresent() && !HelperFunctions.isNoNameTagged(edge)) {
                so.addSuggestion(edge.getTag(NameTag.KEY).get());
            }
        }
    }

}
