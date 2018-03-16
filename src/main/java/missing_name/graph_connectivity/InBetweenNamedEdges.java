package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import util.SpellObject;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class InBetweenNamedEdges {

    // Filter out edges that are reversed versions of each other
    private static Set<EdgeContainer> filterConnectedEdges(Set<Edge> edges, Edge edge) {
        Set<EdgeContainer> filteredConnectedEdges = new HashSet<>();
        EdgeContainer container = new EdgeContainer(edge);
        filteredConnectedEdges.add(container);
        for (Edge e : edges) {
            filteredConnectedEdges.add(new EdgeContainer(e));
        }
        filteredConnectedEdges.remove(container);
        return filteredConnectedEdges;
    }

    public static void run(SpellObject so, Edge edge){
        Set<Edge> connectedInEdges = edge.inEdges();
        Set<Edge> connectedOutEdges = edge.outEdges();

        Set<EdgeContainer> filteredInEdges = filterConnectedEdges(connectedInEdges, edge);
        Set<EdgeContainer> filteredOutEdges = filterConnectedEdges(connectedOutEdges, edge);

        List<String> namesIn = filteredInEdges.stream().map(EdgeContainer::getName).collect(Collectors.toList());
        List<String> namesOut = filteredOutEdges.stream().map(EdgeContainer::getName).collect(Collectors.toList());
        if (namesIn.size() == 1 && namesOut.size() == 1) {
            String name1 = namesIn.get(0).trim().toLowerCase();
            String name2 = namesOut.get(0).trim().toLowerCase();
            if (!name1.isEmpty() && !name2.isEmpty() && name1.equals(name2)) {
                so.addSuggestion(name1);
            }
        }
    }
}
