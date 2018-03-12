package missing_name.graph_connectivity;

import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.HighwayTag;
import org.openstreetmap.atlas.tags.names.NameTag;

import java.util.*;
import java.util.stream.Collectors;


public class nameMerger {
//    private static String ATLAS_FILE = "data/atlas/POINT.atlas";
    private static String ATLAS_FILE = "data/atlas/POINT (8.194954 55.8843184).atlas";
    private static final boolean IN = true;
    private static final boolean OUT = false;

    private static String inBetweenMultipleEdges(Edge edge){
        Set<Long> idsLeft = new HashSet<>();
        Set<Long> idsRight = new HashSet<>();
        idsLeft.add(Math.abs(edge.getIdentifier()));
        idsRight.add(Math.abs(edge.getIdentifier()));
        String nameLeft = propagateEdge(edge.inEdges(), idsLeft);
        String nameRight = propagateEdge(edge.outEdges(), idsRight);
        if(nameLeft.equals(nameRight) && !nameLeft.isEmpty()) return nameLeft;
        return "";
    }

    private static String propagateEdge(Set<Edge> edges, Set<Long> idsChecked){
        Set<Long> idsLeft = new HashSet<>(idsChecked);
        Set<Long> idsRight = new HashSet<>(idsChecked);
        // TODO make sure that idsLeft&idsRight isn't checked. Because they're checked by using DP
        // TODO make check if only one name possible? Propagate others too..!

        Set<String> names = edges.stream()
                .map(it -> it.getTag(NameTag.KEY).orElse(""))
                .collect(Collectors.toSet());
        names.remove("");
        if(names.size() == 1) return new ArrayList<>(names).get(0);
//        if(names.isEmpty()){
//            List<String> recursiveNames = new ArrayList<>();
//            edges.forEach();
//        }
        return "";
    }

    private static String propagatedInBetweenEdges(Edge edge){
        Set<Edge> inEdges = edge.inEdges();
        Set<Edge> outEdges = edge.outEdges();
        inEdges.removeIf(edge::isReversedEdge);
        outEdges.removeIf(edge::isReversedEdge);
        if(inEdges.size() != 1 && outEdges.size() != 1) return "";



        return "";
    }

    private static String smallTailCase(Edge edge){   // TODO improve check to not include noname that leads somewhere!
        Set<Edge> connectedEdges = edge.connectedEdges();
        Set<Long> edgeIds = new HashSet<>();
        connectedEdges.forEach(it -> edgeIds.add(Math.abs(it.getIdentifier())));
        edgeIds.add(Math.abs(edge.getIdentifier()));

        if(edgeIds.size() == 2 && edge.hasReverseEdge()){        // Only return edge & start-edge present.
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
                && inNames.toArray()[0].equals(outNames.toArray()[0]))
            return (String) inNames.toArray()[0];
        else return "";
    }

    private static String changeOfPathType(Edge edge){
        Set<Edge> connectedEdges = edge.connectedEdges();
        String highwayType = edge.getTag(HighwayTag.KEY).orElse("");
        if(highwayType.isEmpty()) return "";

        if(connectedEdges.stream().anyMatch(it -> {
            String itType = it.getTag(HighwayTag.KEY).orElse("");
            return !itType.isEmpty() && itType.equals(highwayType);
        })){
            return "Add noname=true tag";
        }
        return "";
    }

    public static String isRoundAbout(Edge edge){
        if(edge.getTag("junction").orElse("").equals("roundabout")) return "Add noname=true tag";
        return "";
    }

    public static String generateNameSuggestion(Edge edge){     //	motorway_link --> name of motorway
        String name = edge.getTag(NameTag.KEY).orElse("");
        String noname = edge.getTag("noname").orElse("");
        if(name.isEmpty() && noname.isEmpty()){
            String changeOfPathType = changeOfPathType(edge);
//            if(!changeOfPathType.isEmpty()) {
//                System.out.println("changeOfPathType found!");
//                return changeOfPathType;
//            }

            String roundabout = isRoundAbout(edge);
            if(!roundabout.isEmpty()){
                System.out.println("roundabout found!");
                return roundabout;
            }

            String smallTail = smallTailCase(edge);
            if(!smallTail.isEmpty()){
                System.out.println("smallTail found!");
                return smallTail;
            }


            String inBetween = inBetweenNamedEdges(edge);
            if(!inBetween.isEmpty()){
                System.out.println("inBetween found!");
                return inBetween;
            }

            Set<Edge> inEdges = edge.inEdges();
            Set<Edge> outEdges = edge.outEdges();
            System.out.println(edge.getMasterEdgeIdentifier());
            System.out.println(smallTailCase(edge));
            inEdges.forEach(it -> System.out.println("in: " + it.getTag(NameTag.KEY).orElse("(empty)") + ", id: " + it.getMasterEdgeIdentifier()));
            outEdges.forEach(it -> System.out.println("out: " + it.getTag(NameTag.KEY).orElse("(empty)") + ", id: " + it.getMasterEdgeIdentifier()));
            System.out.println("=======================================================");
        }

        return "";
    }

    public static void main(String[] args) {
        Atlas atlas = new AtlasResourceLoader().load(new File(ATLAS_FILE));
        int i = 0, j = 0;
        for(Edge edge: atlas.edges()){
            String suggestion = generateNameSuggestion(edge);
            i++;
            j++;
            if(!suggestion.isEmpty()) System.out.println("Suggestion: " + suggestion + " (" + edge.getIdentifier() + ")");
            else j--;
        }
        System.out.println("Edited " + j + " items (of " + i + ")");
    }
}
