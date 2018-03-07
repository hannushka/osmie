package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.names.NameTag;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collector;
import java.util.stream.Collectors;


public class nameMerger {
    private static String ATLAS_FILE = "data/atlas/POINT (8.194954 55.8843184).atlas";

    private static String smallTailCase(Edge edge){   // Also check noname-tag.
        Set<Edge> connectedEdges = edge.connectedEdges();
        Optional<Edge> reverseEdge = edge.reversed();
        Set<Long> uniqueIdsConnected = new HashSet<>();
        for(Edge edgeIn : connectedEdges) uniqueIdsConnected.add(edgeIn.getMasterEdgeIdentifier());
        if(reverseEdge.isPresent() && connectedEdges.contains(reverseEdge.get()) && uniqueIdsConnected.size() == 2){
            String name = "";
            for(Edge edgeConnect: connectedEdges) name = edgeConnect.getTag(NameTag.KEY).orElse(name);
            return name;
        }else{
            return "";
        }
    }

    private static String inBetweenNamedEdges(Edge edge){      // Starting with ONLY the case of MASVAGEN -> __ -> MASVAGEN
        Set<Edge> inEdges = edge.inEdges();
        Set<Edge> outEdges = edge.outEdges();
        inEdges.removeIf(it -> it.isReversedEdge(edge));
        outEdges.removeIf(it -> it.isReversedEdge(edge));
        Set<String> inNames = inEdges.stream()
                .map(it -> it.getTag(NameTag.KEY).orElse(""))
                .collect(Collectors.toSet());
        inNames.remove("");
        Set<String> outNames = inEdges.stream()
                .map(it -> it.getTag(NameTag.KEY).orElse(""))
                .collect(Collectors.toSet());
        outNames.remove("");
        if(inNames.size() == 1 && outNames.size() == 1 && inNames.toArray()[0].equals(outNames.toArray()[0]))
            return (String) inNames.toArray()[0];
        else return "";
    }

    public static String generateNameSuggestion(Edge edge){
        String name = edge.getTag(NameTag.KEY).orElse("");
        String noname = edge.getTag("noname").orElse("");
        if(name.isEmpty() && noname.isEmpty()){
            String smallTail = smallTailCase(edge);
            if(!smallTail.isEmpty()) return smallTail;

            System.out.println("Going for inBet");
            String inBetween = inBetweenNamedEdges(edge);
            if(!inBetween.isEmpty()) return inBetween;
            System.out.println("Fuck.");

            Set<Edge> inEdges = edge.inEdges();
            Set<Edge> outEdges = edge.outEdges();
            System.out.println(edge.getMasterEdgeIdentifier());
            System.out.println(smallTailCase(edge));
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
