import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.names.NameTag;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class Main {

    private String tmpAtlasFile = "data/atlas/POINT (7.9321289 55.5783983)=POINT (8.0998535 55.6902135).atlas";
    private String tmpOldAtlasFile = "data/atlas_old/POINT (7.9321289 55.5783983)=POINT (8.0998535 55.6902135).atlas";

    public void run() {
        final Atlas atlasLoad = new AtlasResourceLoader().load(new File(tmpAtlasFile));
        final Atlas atlasOldLoad = new AtlasResourceLoader().load(new File(tmpOldAtlasFile));

        Map<String, String> names = new HashMap();
        atlasLoad.edges().forEach((Edge edge) -> {
            Optional<Edge> newEdge = Optional.ofNullable(edge);
            Optional<Edge> oldEdge = Optional.ofNullable(atlasOldLoad.edge(edge.getIdentifier()));
            if (oldEdge.isPresent() && oldEdge.get().getTag(NameTag.KEY).isPresent() &&
                    newEdge.isPresent() && newEdge.get().getTag(NameTag.KEY).isPresent()) {
                String oldEdgeName = oldEdge.get().getTag(NameTag.KEY).get();
                String newEdgeName = newEdge.get().getTag(NameTag.KEY).get();
                names.put(oldEdgeName, newEdgeName);
            }
        });
//        names.forEach((k,v) -> System.out.println(k + "-" + v));
        WeightingAlgorithm.runAlgorithm("Pallesdamvej", "Pallesdamvej");
    }

    public static void main(String[] args) {
        new Main().run();
    }
}
