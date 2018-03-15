package missing_name.graph_connectivity;

import org.jetbrains.annotations.NotNull;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import util.EdgeContainer;
import util.SpellObject;

import java.util.*;
import java.util.stream.Collectors;

public class GraphConnectivityCorrector {

    public static void run(SpellObject so, Edge e) {
        Optional<String> name = e.getTag("name");
        if (!name.isPresent()) {
            inBetweenNamedEdges(so, e);
        }
    }

    private static void inBetweenNamedEdges(SpellObject so, Edge edge){
        Set<Edge> connectedInEdges = edge.inEdges();
        Set<Edge> connectedOutEdges = edge.outEdges();

        // Filter out edges that are reversed versions of each other
        Set<EdgeContainer> filteredInConnectedEdges = new HashSet<>();
        EdgeContainer container = new EdgeContainer(edge);
        filteredInConnectedEdges.add(container);
        for (Edge e : connectedInEdges) {
            filteredInConnectedEdges.add(new EdgeContainer(e));
        }
        filteredInConnectedEdges.remove(container);

        Set<EdgeContainer> filteredOutConnectedEdges = new HashSet<>();
        filteredOutConnectedEdges.add(container);
        for (Edge e : connectedOutEdges) {
            filteredOutConnectedEdges.add(new EdgeContainer(e));
        }
        filteredOutConnectedEdges.remove(container);

        List<String> namesIn = filteredInConnectedEdges.stream().map(EdgeContainer::getName).collect(Collectors.toList());
        List<String> namesOut = filteredOutConnectedEdges.stream().map(EdgeContainer::getName).collect(Collectors.toList());
        if (namesIn.size() == 1 && namesOut.size() == 1) {
            String name1 = namesIn.get(0).trim().toLowerCase();
            String name2 = namesOut.get(0).trim().toLowerCase();
            if (!name1.isEmpty() && !name2.isEmpty() && name1.equals(name2)) {
                so.addSuggestion(name1);
            }
        }
    }
}
