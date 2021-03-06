import org.openstreetmap.atlas.geography.Location;
import org.openstreetmap.atlas.geography.MultiPolygon;
import org.openstreetmap.atlas.geography.Rectangle;
import org.openstreetmap.atlas.geography.atlas.Atlas;
import org.openstreetmap.atlas.geography.atlas.AtlasResourceLoader;
import org.openstreetmap.atlas.geography.atlas.items.Edge;
import org.openstreetmap.atlas.geography.atlas.pbf.AtlasLoadingOption;
import org.openstreetmap.atlas.geography.atlas.pbf.OsmPbfLoader;
import org.openstreetmap.atlas.streaming.resource.File;
import org.openstreetmap.atlas.tags.*;
import org.openstreetmap.atlas.tags.names.NameTag;
import symspell.SymSpell;
import util.EditDistance;

import java.io.*;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Setup {
    private static void download(String url, String fileName) {
        try (InputStream in = URI.create(url).toURL().openStream()) {
            Files.copy(in, Paths.get(fileName));
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    private static void extractNameData(BufferedWriter writer, final File atlasFile, final File atlasOld) throws IOException{
        final Atlas atlasLoad = new AtlasResourceLoader().load(atlasFile);
        final Atlas atlasOldLoad = new AtlasResourceLoader().load(atlasOld);
        String newEdgeName, oldEdgeName, speed;
        Optional<Edge> newEdge;
        for(Edge oldEdge: atlasOldLoad.edges()){
            if(!oldEdge.isMasterEdge()) continue;

            newEdge = Optional.ofNullable(atlasLoad.edge(oldEdge.getIdentifier()));
            if(newEdge.isPresent()){
                if(newEdge.get().getTag(NameTag.KEY).isPresent() && oldEdge.getTag(NameTag.KEY).isPresent()){
                    newEdgeName = newEdge.get().getTag(NameTag.KEY).get();
                    oldEdgeName = oldEdge.getTag(NameTag.KEY).get();
                    speed = newEdge.get().getTag(MaxSpeedTag.KEY).orElse("-1");
                    writer.write(oldEdgeName + ",,," + newEdgeName + ",,," + speed + "\n");
                }
            }
        }
    }

    private static void extractData(BufferedWriter writer, final File atlasFile, final File atlasOld) throws IOException{
        final Atlas atlasLoad = new AtlasResourceLoader().load(atlasFile);
        final Atlas atlasOldLoad = new AtlasResourceLoader().load(atlasOld);
        StringJoiner joiner;
        Optional<Edge> newEdgeOpt;
        Edge newEdge;
        for(Edge oldEdge: atlasOldLoad.edges()){
            if(!oldEdge.isMasterEdge()) continue;

            newEdgeOpt = Optional.ofNullable(atlasLoad.edge(oldEdge.getIdentifier()));
            if(newEdgeOpt.isPresent()){
                if(newEdgeOpt.get().getTag(NameTag.KEY).isPresent() && oldEdge.getTag(NameTag.KEY).isPresent()){
                    joiner = new StringJoiner(",,,");
                    newEdge = newEdgeOpt.get();

                    joiner.add(oldEdge.getTag(NameTag.KEY).get());
//                    joiner.add(newEdge.getTag(MaxSpeedTag.KEY).orElse("-1"));
//                    joiner.add(newEdge.getTag(SurfaceTag.KEY).orElse("-1"));
//                    joiner.add(newEdge.getTag(HighwayTag.KEY).orElse("-1"));
//                    joiner.add(newEdge.getTag(SidewalkTag.KEY).orElse("-1"));
//                    joiner.add(newEdge.getTag(OneWayTag.KEY).orElse("-1"));
                    joiner.add(newEdge.getTag(NameTag.KEY).get());

                    writer.write(joiner.toString() + "\n");
                }
            }
        }
    }

    private static void extractNameDataFromAtlasFiles(String ATLAS_FOLDER, String ATLAS_OLD_FOLDER, String NAMEDATA_FILE) {
        java.io.File folder = new java.io.File(ATLAS_FOLDER);
        java.io.File[] listOfFiles = folder.listFiles();
        try {
            final FileWriter fw = new FileWriter(NAMEDATA_FILE);
            final BufferedWriter writer = new BufferedWriter(fw);
            String filename;
            File atlasFile, atlasFileOld;
            assert listOfFiles != null;
            for (int i = 0; i < listOfFiles.length; i++) {
                filename = listOfFiles[i].getName();
                System.out.println("Processing file " + i);
                atlasFile = new File(ATLAS_FOLDER + "/" + filename);
                atlasFileOld = new File(ATLAS_OLD_FOLDER + "/" + filename);
                if (atlasFile.exists() && atlasFileOld.exists()) extractNameData(writer, atlasFile, atlasFileOld);
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void removeLargeChanges(String from, String to) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(to));
        try (BufferedReader br = Files.newBufferedReader(Paths.get(from))) {
            for(String line; (line = br.readLine()) != null;){
//                if(line.isEmpty()) continue;
                String[] split = line.split(",,,");
//                if(split.length < 2) continue;
                String newWord = split[1];
                EditDistance editDistance = new EditDistance(newWord.toLowerCase().trim());
                if(editDistance.DamerauLevenshteinDistance(split[0].toLowerCase().trim(), 3) != -1)
                    writer.write(line + '\n');
            }
        }
    }

    private static void filterNameDataToUnique(String from, String to) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(to));
        BufferedWriter trainWriter = new BufferedWriter(new FileWriter("data/trainData.csv"));
        BufferedWriter testWriter = new BufferedWriter(new FileWriter("data/testData.csv"));

        Set<String> streetChanges = new HashSet<>();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(from))) {
            for(String line; (line = br.readLine()) != null;){
                streetChanges.add(line);
            }
        }
        System.out.println(streetChanges.size());
        streetChanges.forEach(s -> {
            try {
                writer.write(s + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        ArrayList<String> streetList = new ArrayList<>(streetChanges);
        int split = (int)(streetList.size() * 2f/3);
        for(int i = 0; i < split; i++) trainWriter.write(streetList.get(i) + "\n");
        for(int i = split; i < streetList.size(); i++) testWriter.write(streetList.get(i) + "\n");

        writer.close();
        trainWriter.close();
        testWriter.close();
    }

    private static void filterSuperData(String from, String to) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(to));

        Set<String> streetChanges = new HashSet<>();
        try (BufferedReader br = Files.newBufferedReader(Paths.get(from))) {
            for(String line; (line = br.readLine()) != null;){
                streetChanges.add(line);
            }
        }
        System.out.println(streetChanges.size());
        streetChanges.forEach(s -> {
            try {
                writer.write(s + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        writer.close();
    }

    private static void introduceNoiseToNameData(){
        ProcessBuilder pb = new ProcessBuilder("python3", "scripts/auto_noise.py");
        try {
            pb.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void extractDataFromAtlasFiles(String ATLAS_FOLDER, String ATLAS_OLD_FOLDER, String NAMEDATA_FILE) {
        java.io.File folder = new java.io.File(ATLAS_FOLDER);
        java.io.File[] listOfFiles = folder.listFiles();
        try {
            final FileWriter fw = new FileWriter(NAMEDATA_FILE);
            final BufferedWriter writer = new BufferedWriter(fw);
            String filename;
            File atlasFile, atlasFileOld;
            assert listOfFiles != null;
            for (int i = 0; i < listOfFiles.length; i++) {
                filename = listOfFiles[i].getName();
                System.out.println("Processing file " + i);
                atlasFile = new File(ATLAS_FOLDER + "/" + filename);
                atlasFileOld = new File(ATLAS_OLD_FOLDER + "/" + filename);
                if (atlasFile.exists() && atlasFileOld.exists()) extractNameData(writer, atlasFile, atlasFileOld);
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void filterManual(String file) throws IOException {
        StringJoiner joiner = new StringJoiner("\n");
        String[] split;
        try (BufferedReader br = Files.newBufferedReader(Paths.get(file))) {
            for(String line; (line = br.readLine()) != null;){
                split = line.split(",,,");
                if(new EditDistance(split[0]).DamerauLevenshteinDistance(split[1],3) != -1)
                    joiner.add(line);
            }
        }
        BufferedWriter writer = new BufferedWriter(new FileWriter("data/filteredManualNameData.csv"));
        writer.write(joiner.toString());
        writer.flush();
        writer.close();
    }

    private static ArrayList<Rectangle> getQuadrupleFromZone(Rectangle rec){
        Location lowerLeft = rec.lowerLeft();
        Location lowerRight = rec.lowerRight();
        Location upperRight = rec.upperRight();
        Location upperLeft = rec.upperLeft();
        ArrayList<Rectangle> zones = new ArrayList<>(4);
        zones.add(Rectangle.forCorners(lowerLeft, lowerLeft.midPoint(upperRight)));
        zones.add(Rectangle.forCorners(lowerLeft.midPoint(lowerRight), lowerRight.midPoint(upperRight)));
        zones.add(Rectangle.forCorners(lowerLeft.midPoint(upperLeft), upperLeft.midPoint(upperRight)));
        zones.add(Rectangle.forCorners(lowerLeft.midPoint(upperRight), upperRight));
        return zones;
    }

    private static void writeAtlas(String filename, String resultpath, List<MultiPolygon> zones,
                                   AtlasLoadingOption option){
        OsmPbfLoader loader;
        Atlas atlas;
        for(int i = 0; i < zones.size(); i++){
            MultiPolygon zone = zones.get(i);
            loader = new OsmPbfLoader(new File(filename), zone, option);
            atlas = loader.read();
            if(atlas != null && atlas.numberOfEdges() > 10) atlas.save(new File(String.format("%s/%s.atlas",
                    resultpath, zone.bounds().center())));
        }
    }

    private static ArrayList<Rectangle> getQuadrupleFromZones(List<Rectangle> rects){
        ArrayList<Rectangle> quadruples = new ArrayList<>(rects.size() * 4);
        rects.forEach(it -> quadruples.addAll(getQuadrupleFromZone(it)));
        return quadruples;
    }

    public static void main(String[] args) throws IOException {
        String OSM_PBF_15 = "http://download.geofabrik.de/europe/denmark-150101.osm.pbf";
        String OSM_PBF_LATEST = "http://download.geofabrik.de/europe/denmark-latest.osm.pbf";
        String OSM_PBF_LOCAL_15 = "data/denmark-150101.osm.pbf";
        String OSM_PBF_LOCAL_LATEST = "data/denmark-latest.osm.pbf";
        String ATLAS_FOLDER = "data/et_atlas";
        String ATLAS_OLD_FOLDER = "data/et_atlas_old";
        String NAMEDATA_FILE = "data/et_nameData.csv";
        String DATA_FILE = "data/et_SuperData.csv";
        String DATA_UNIQUE_FILE = "data/et_SuperDataUnique.csv";
        String NAMEDATA_FIXED_FILE = "data/et_nameDataNew.csv";
        String NAMEDATA_UNIQUE_FILE = "data/et_nameDataUnique.csv";
//        download(OSM_PBF_15, OSM_PBF_LOCAL_15);
//        download(OSM_PBF_LATEST, OSM_PBF_LOCAL_LATEST);
        // TODO introduce atlas generation here
//        AtlasLoadingOption option = AtlasLoadingOption.withNoFilter();
//        option
//                .setLoadWaysSpanningCountryBoundaries(false)
//                .setLoadAtlasPoint(false)
//                .setLoadAtlasLine(false)
//                .setLoadAtlasArea(false)
//                .setLoadAtlasRelation(false)
//                .setWaySectioning(false);
//        Rectangle est = Rectangle.forString("57.494,21.377:59.946,28.229");
//        List<MultiPolygon> est_mp = getQuadrupleFromZones(getQuadrupleFromZone(est)).stream()
//                                        .map(MultiPolygon::forPolygon)
//                                        .collect(Collectors.toList());
//        writeAtlas("data/estonia-140101.osm.pbf", "data/et_atlas_old", est_mp, option);
//        extractDataFromAtlasFiles(ATLAS_FOLDER, ATLAS_OLD_FOLDER, NAMEDATA_FILE);
//        filterSuperData(DATA_FILE, DATA_UNIQUE_FILE);
//        removeLargeChanges(NAMEDATA_FILE, NAMEDATA_FIXED_FILE);
//        filterNameDataToUnique(NAMEDATA_FIXED_FILE, NAMEDATA_UNIQUE_FILE);
//        introduceNoiseToNameData();
    }
}
