package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.names.NameTag;

import java.util.HashSet;
import java.util.Set;


public class nameMerger {
    private static String ATLAS_FILE = "data/atlas/POINT (8.194954 55.8843184).atlas";

    private static String getName(Set<Edge> edges){
        String name = "";
        for(Edge edge: edges) name = edge.getTag(NameTag.KEY).orElse(name);
        return name;
    }

    private static boolean tailCaseNoName(Edge edge, Set<Edge> inEdges, Set<Edge> outEdges){   // Also check noname-tag.
        Set<Long> uniqueIdsConnected = new HashSet<>();
        boolean noName = !edge.getTag("noname").isPresent();
        for(Edge edgeIn : inEdges) uniqueIdsConnected.add(edgeIn.getMasterEdgeIdentifier());
        for(Edge edgeOut : outEdges) uniqueIdsConnected.add(edgeOut.getMasterEdgeIdentifier());
        return edge.hasReverseEdge() && uniqueIdsConnected.size() == 2 && noName;  // Will contain itself if reversed!
    }

    private static boolean inBetweenNamedEdges(){

        return false;
    }

    public static String generateNameSuggestion(Edge edge){
        String name = edge.getTag(NameTag.KEY).orElse("");
        if(name.isEmpty()){
            Set<Edge> inEdges = edge.inEdges();
            Set<Edge> outEdges = edge.outEdges();
            if(tailCaseNoName(edge, inEdges, outEdges)){
                return getName(edge.connectedEdges());
            }
            if(inBetweenNamedEdges()){
                // TODO implement this
            }

            System.out.println(edge.getMasterEdgeIdentifier());
            System.out.println(tailCaseNoName(edge, inEdges, outEdges));
            inEdges.forEach(it -> System.out.println("in: " + it.getTag(NameTag.KEY).orElse("(empty)") + ", id: " + it.getMasterEdgeIdentifier()));
            outEdges.forEach(it -> System.out.println("out: " + it.getTag(NameTag.KEY).orElse("(empty)") + ", id: " + it.getMasterEdgeIdentifier()));
            System.out.println("=======================================================");
        }

        return name;
    }

    public static void main(String[] args) {
        Atlas atlas = new AtlasResourceLoader().load(new File(ATLAS_FILE));
        for(Edge edge: atlas.edges()){
            String suggestion = generateNameSuggestion(edge);
            System.out.println("Suggestion: " + suggestion);
        }
    }
}
