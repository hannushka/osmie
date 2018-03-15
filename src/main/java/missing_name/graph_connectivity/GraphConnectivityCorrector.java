package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.tags.names.NameTag;
import util.SpellObject;

import java.util.Set;
import java.util.stream.Collectors;

public class GraphConnectivityCorrector {

    public static void run(SpellObject so, Edge e) {
        inBetweenNamedEdges(so, e);
    }

    private static  void inBetweenNamedEdges(SpellObject so, Edge edge){
        Set<Edge> inEdges = edge.inEdges();
        Set<Edge> outEdges = edge.outEdges();
        inEdges.removeIf(it -> it.isReversedEdge(edge) || it.connectedEdges().stream().anyMatch(inEdges::contains));
        outEdges.removeIf(it -> it.isReversedEdge(edge) || it.connectedEdges().stream().anyMatch(outEdges::contains));
        Set<String> inNames = inEdges.stream()
                .map(it -> it.getTag(NameTag.KEY).orElse(""))
                .collect(Collectors.toSet());
        inNames.remove("");
        Set<String> outNames = inEdges.stream()
                .map(it -> it.getTag(NameTag.KEY).orElse(""))
                .collect(Collectors.toSet());
        outNames.remove("");
        if(inNames.size() == 1 && outNames.size() == 1
                && inNames.toArray()[0].equals(outNames.toArray()[0])) {
            String sugg = (String)inNames.toArray()[0];
            so.addSuggestion(sugg.toLowerCase());
        }
    }
}
