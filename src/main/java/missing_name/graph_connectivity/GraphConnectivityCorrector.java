package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.items.Edge;
import util.HelperFunctions;
import util.SpellObject;

import java.util.*;

public class GraphConnectivityCorrector {

    public static void run(SpellObject so, Edge edge) {
        if (HelperFunctions.isNoNameTagged(edge))
            return;
        Optional<String> name = edge.getTag("name");
        if (!name.isPresent() && !HelperFunctions.isRoundAbout(edge)) {
            Set<Edge> connectedInEdges = edge.inEdges();
            Set<Edge> connectedOutEdges = edge.outEdges();
            Set<EdgeContainer> filteredInEdges = filterConnectedEdges(connectedInEdges, edge);
            Set<EdgeContainer> filteredOutEdges = filterConnectedEdges(connectedOutEdges, edge);
            // Check simple smoothing case
            SmoothingCases.inBetweenNamedEdgesCheck(filteredInEdges, filteredOutEdges, so);
            // Check dangling segment cases
            SmoothingCases.tailCaseCheck(filteredInEdges, so);
        }
    }

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
}
