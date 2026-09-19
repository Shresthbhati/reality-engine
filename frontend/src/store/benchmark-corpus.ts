import type { BenchmarkStructure } from '@/types/reality-engine';

/**
 * Reality Engine — Engineering Benchmark Corpus
 * 
 * Curated architectural engineering benchmark corpus for spatial reconstruction stress tests.
 * This is an engineering validation suite, NOT an authoritative ranking.
 *
 * Grounded Reality Statuses:
 * - Victoria Memorial (euro-009): VALIDATED with LOCAL_DATASET and COMPILED WorldIR.
 * - Others: Honestly marked with real states (EVIDENCE_AVAILABLE, AERIAL_ARCHIVE, NOT_YET_CAPTURED, etc.).
 */
export const BENCHMARK_STRUCTURES: BenchmarkStructure[] = [
  {
    "id": "skysc-001",
    "name": "Burj Khalifa",
    "category": "SKYSCRAPER",
    "location": "Dubai",
    "country": "United Arab Emirates",
    "scale": "828.0 m (163 floors)",
    "complexity": "EXTREME",
    "architectureTypology": "Buttressed Core Supertall",
    "era": "Modern (2010)",
    "heightMeters": 828,
    "footprintSqMeters": 88000,
    "captureTypes": [
      "HIGH_ALT_DRONE",
      "AERIAL_LIDAR",
      "GROUND_SURVEY"
    ],
    "expectedImages": 28500,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "CAD_SURVEY",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Y-shaped tri-axial buttressed core stepping back in 27 tiers to a central spire.",
    "keyChallenges": [
      "Severe atmospheric dust and thermal shimmer at 800m+",
      "Specular solar glazing causing multi-view photogrammetry outliers",
      "Vertical GNSS altitude dilution of precision (PDOP > 4.2)"
    ],
    "benchmarkObjectives": [
      "Sub-centimeter vertical coordinate drift over 800 vertical meters",
      "Specular reflection rejection on glass curtain walls",
      "High-altitude aerodynamic geometry extraction"
    ]
  },
  {
    "id": "skysc-002",
    "name": "Merdeka 118",
    "category": "SKYSCRAPER",
    "location": "Kuala Lumpur",
    "country": "Malaysia",
    "scale": "678.9 m (118 floors)",
    "complexity": "EXTREME",
    "architectureTypology": "Faceted Mega-tall with Spire",
    "era": "Modern (2023)",
    "heightMeters": 678.9,
    "footprintSqMeters": 72000,
    "captureTypes": [
      "DRONE_ORBIT",
      "AERIAL_LIDAR"
    ],
    "expectedImages": 24000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Diamond-shaped faceted glass facade inspired by Malaysian songket motifs with 160m spire.",
    "keyChallenges": [
      "Non-orthogonal diamond facet triangular glass reflections",
      "Off-center slender spire structural occlusion"
    ],
    "benchmarkObjectives": [
      "Diamond facet normal field reconstruction",
      "High-aspect-ratio slender spire point cloud density"
    ]
  },
  {
    "id": "skysc-003",
    "name": "Shanghai Tower",
    "category": "SKYSCRAPER",
    "location": "Shanghai",
    "country": "China",
    "scale": "632.0 m (128 floors)",
    "complexity": "EXTREME",
    "architectureTypology": "Twisted Double-Skin Tower",
    "era": "Modern (2015)",
    "heightMeters": 632,
    "footprintSqMeters": 65000,
    "captureTypes": [
      "AERIAL_OBLIQUE",
      "HIGH_RES_RGB"
    ],
    "expectedImages": 26000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "120-degree continuous twist around a cylindrical core with transparent double-curtain glass wall.",
    "keyChallenges": [
      "Double-skin glass facade with internal atriums visible through outer wall",
      "Continuous non-linear 120° rotational twist"
    ],
    "benchmarkObjectives": [
      "Reconstruction of double-curtain wall inner and outer boundary surfaces",
      "Rotational symmetry pose estimation under marine humidity"
    ]
  },
  {
    "id": "skysc-004",
    "name": "Makkah Royal Clock Tower",
    "category": "SKYSCRAPER",
    "location": "Mecca",
    "country": "Saudi Arabia",
    "scale": "601.0 m (120 floors)",
    "complexity": "EXTREME",
    "architectureTypology": "Composite Skyscraper & Mega-Clock",
    "era": "Post-Modern (2012)",
    "heightMeters": 601,
    "footprintSqMeters": 92000,
    "captureTypes": [
      "AERIAL_PHOTOGRAMMETRY"
    ],
    "expectedImages": 22000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Massive 43m clock faces mounted on structural steel frame topped with golden crescent.",
    "keyChallenges": [
      "Intense desert sun glare",
      "Restricted airspace limiting low-altitude flight paths"
    ],
    "benchmarkObjectives": [
      "Mosaic glass dial surface extraction at 400m elevation",
      "Crescent finial reconstruction from long-range oblique sensors"
    ]
  },
  {
    "id": "skysc-005",
    "name": "Ping An Finance Center",
    "category": "SKYSCRAPER",
    "location": "Shenzhen",
    "country": "China",
    "scale": "599.1 m (115 floors)",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Tapered Stainless-Steel Composite",
    "era": "Modern (2017)",
    "heightMeters": 599.1,
    "footprintSqMeters": 61000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY"
    ],
    "expectedImages": 19500,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Four outward-curving corner mega-columns with 1,700 tons of 316L stainless steel ribs.",
    "keyChallenges": [
      "Brushed stainless steel chevron reflections",
      "Narrow urban canyon interference between adjacent high-rises"
    ],
    "benchmarkObjectives": [
      "Metallic bidirectional reflectance distribution function (BRDF) handling",
      "Corner mega-column verticality validation"
    ]
  },
  {
    "id": "skysc-006",
    "name": "Lotte World Tower",
    "category": "SKYSCRAPER",
    "location": "Seoul",
    "country": "South Korea",
    "scale": "555.0 m (123 floors)",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Conical Tapered Obelisk",
    "era": "Modern (2017)",
    "heightMeters": 555,
    "footprintSqMeters": 58000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 18000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Softly curving tapered profile inspired by traditional Korean ceramics and calligraphy brushes.",
    "keyChallenges": [
      "Gentle continuous convex curvature without hard geometric edges",
      "River surface reflection from adjacent Han river"
    ],
    "benchmarkObjectives": [
      "Curvature continuity extraction without polyhedral facet collapse"
    ]
  },
  {
    "id": "skysc-007",
    "name": "One World Trade Center",
    "category": "SKYSCRAPER",
    "location": "New York City",
    "country": "United States",
    "scale": "541.3 m (104 floors)",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Antiprismatic Tapered Monolith",
    "era": "Modern (2014)",
    "heightMeters": 541.3,
    "footprintSqMeters": 48000,
    "captureTypes": [
      "AERIAL_PHOTOGRAMMETRY",
      "MUNICIPAL_LIDAR"
    ],
    "expectedImages": 21000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Square base transitions through eight elongated isosceles triangles into an octagonal midsection.",
    "keyChallenges": [
      "Dense Manhattan GPS multipath bounce from neighboring glass towers",
      "Prismatic tapering facet boundary detection"
    ],
    "benchmarkObjectives": [
      "Urban multipath GNSS rejection using visual odometry bundle adjustment",
      "Isosceles triangle sharp edge preservation"
    ],
    "evidenceSummary": {
      "photos": 4200,
      "datasetSizeGb": 64.2,
      "sourceUri": "archive://us-ny-manhattan-open/wtc1_aerial_2024"
    }
  },
  {
    "id": "skysc-008",
    "name": "Guangzhou CTF Finance Centre",
    "category": "SKYSCRAPER",
    "location": "Guangzhou",
    "country": "China",
    "scale": "530.0 m (111 floors)",
    "complexity": "HIGH",
    "architectureTypology": "Terraced Setback High-Rise",
    "era": "Modern (2016)",
    "heightMeters": 530,
    "footprintSqMeters": 50000,
    "captureTypes": [
      "DRONE_RGB"
    ],
    "expectedImages": 17500,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Stepped setbacks with glazed terracotta vertical mullions.",
    "keyChallenges": [
      "Glazed terracotta vertical mullion shadow casting",
      "Atmospheric humidity scattering"
    ],
    "benchmarkObjectives": [
      "Terracotta vs glass material boundary classification"
    ]
  },
  {
    "id": "skysc-009",
    "name": "Tianjin CTF Finance Centre",
    "category": "SKYSCRAPER",
    "location": "Tianjin",
    "country": "China",
    "scale": "530.0 m (97 floors)",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Curvilinear Aerodynamic Tower",
    "era": "Modern (2019)",
    "heightMeters": 530,
    "footprintSqMeters": 45000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY"
    ],
    "expectedImages": 16900,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Curvilinear aerodynamic envelope with recessed vent openings to minimize vortex shedding.",
    "keyChallenges": [
      "Multi-axis non-repeating curvature",
      "Variable sun incidence angles"
    ],
    "benchmarkObjectives": [
      "Aerodynamic vent opening depth estimation"
    ]
  },
  {
    "id": "skysc-010",
    "name": "CITIC Tower (China Zun)",
    "category": "SKYSCRAPER",
    "location": "Beijing",
    "country": "China",
    "scale": "528.0 m (108 floors)",
    "complexity": "HIGH",
    "architectureTypology": "Hyperboloid Concave Vessel",
    "era": "Modern (2018)",
    "heightMeters": 528,
    "footprintSqMeters": 47000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY"
    ],
    "expectedImages": 15500,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Hourglass hyperbolic profile inspired by the ancient Chinese ceremonial 'zun' vessel.",
    "keyChallenges": [
      "Narrowing waist feature tracking across oblique angles",
      "Seasonal Beijing atmospheric particulates"
    ],
    "benchmarkObjectives": [
      "Hyperbolic curvature continuity across narrow waist"
    ]
  },
  {
    "id": "fort-001",
    "name": "Chittorgarh Fort",
    "category": "INDIAN_FORT",
    "location": "Chittorgarh, Rajasthan",
    "country": "India",
    "scale": "2.8 km² plateau (180m hill)",
    "complexity": "EXTREME",
    "architectureTypology": "Hilltop Bastioned Citadel",
    "era": "7th - 16th Century",
    "heightMeters": 180,
    "footprintSqMeters": 2800000,
    "captureTypes": [
      "LONG_RANGE_DRONE",
      "TERRESTRIAL_LIDAR",
      "HANDHELD_PHONE"
    ],
    "expectedImages": 45000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "PARTIAL_CAPTURE",
    "groundTruthAvailability": "SATELLITE_DEM",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "65 historic structures including Vijay Stambha, Kirti Stambha, 4 palace complexes, 19 temples.",
    "keyChallenges": [
      "Vast 2.8 square-kilometer geographic extent",
      "Extensive sheer cliff faces and ruined stone masonry with identical sandstone textures",
      "Harsh desert solar shadows across massive battlements"
    ],
    "benchmarkObjectives": [
      "Multi-kilometer georeferencing without ground loop closure distortion",
      "Vijay Stambha 9-story carved stone relief separation",
      "Cliff-to-fortification contact interface extraction"
    ],
    "evidenceSummary": {
      "photos": 6840,
      "datasetSizeGb": 98.4,
      "sourceUri": "archive://in-raj-chittor-2025"
    }
  },
  {
    "id": "fort-002",
    "name": "Kumbhalgarh Fort",
    "category": "INDIAN_FORT",
    "location": "Rajsamand, Rajasthan",
    "country": "India",
    "scale": "36 km perimeter wall (2nd longest)",
    "complexity": "EXTREME",
    "architectureTypology": "Ridge-line Mountain Fortification",
    "era": "15th Century",
    "heightMeters": 1100,
    "footprintSqMeters": 4500000,
    "captureTypes": [
      "DRONE_TERRAIN_GRID",
      "WALKTHROUGH_SLAM"
    ],
    "expectedImages": 52000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "SATELLITE_DEM",
    "worldIRStatus": "PENDING",
    "structuralNotes": "36 km serpentine mountain wall with 7 fortified gateways across 13 Aravali mountain peaks.",
    "keyChallenges": [
      "Serpentine wall tracking along jagged ridgelines spanning tens of kilometers",
      "Variable mountain vegetation encroaching on ramparts"
    ],
    "benchmarkObjectives": [
      "36km continuous wall topological graph extraction",
      "Vegetation vs stone rampart segmentation"
    ]
  },
  {
    "id": "fort-003",
    "name": "Ranthambore Fort",
    "category": "INDIAN_FORT",
    "location": "Sawai Madhopur, Rajasthan",
    "country": "India",
    "scale": "Hilltop complex within National Park",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Forest Hilltop Bastion",
    "era": "10th Century",
    "heightMeters": 215,
    "footprintSqMeters": 1200000,
    "captureTypes": [
      "DRONE",
      "HANDHELD_CAMERA"
    ],
    "expectedImages": 31000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Precipitous plateau fortress with 7 km battlements, Toran Dwar, and stepwells.",
    "keyChallenges": [
      "Dense dry deciduous jungle canopy hiding lower ramparts",
      "Steep gorge drop-offs"
    ],
    "benchmarkObjectives": [
      "Canopy penetration depth reconstruction",
      "Stepwell subterranean volume extraction"
    ]
  },
  {
    "id": "fort-004",
    "name": "Gagron Fort",
    "category": "INDIAN_FORT",
    "location": "Jhalawar, Rajasthan",
    "country": "India",
    "scale": "Hill and water fort (Jal Durg)",
    "complexity": "VERY_HIGH",
    "architectureTypology": "River Confluence Water Citadel",
    "era": "12th Century",
    "heightMeters": 65,
    "footprintSqMeters": 420000,
    "captureTypes": [
      "DRONE_WATER_SURVEY",
      "BOAT_SCAN"
    ],
    "expectedImages": 24000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Surrounded on three sides by Ahu and Kali Sindh rivers without foundation trench.",
    "keyChallenges": [
      "Moving river water reflections and waves lapping against bastions",
      "Water level boundary ambiguities"
    ],
    "benchmarkObjectives": [
      "Specular river surface masking and waterline elevation determination"
    ]
  },
  {
    "id": "fort-005",
    "name": "Amber Fort (Amer)",
    "category": "INDIAN_FORT",
    "location": "Jaipur, Rajasthan",
    "country": "India",
    "scale": "Multi-tiered hillside palace & fort",
    "complexity": "EXTREME",
    "architectureTypology": "Rajput-Mughal Hilltop Palace",
    "era": "16th Century",
    "heightMeters": 140,
    "footprintSqMeters": 380000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 38000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "CAD_SURVEY",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Tiered yellow-and-pink sandstone and marble courtyards with Sheesh Mahal mirror mosaics and Maota Lake.",
    "keyChallenges": [
      "Intricate mirror-mosaic Sheesh Mahal interior reflection chaos",
      "Multi-tiered courtyards with deep shadowed arcades"
    ],
    "benchmarkObjectives": [
      "Reflective concave convex mirror mosaic surface reconstruction",
      "Jaali screen pierced stonework depth resolution"
    ],
    "evidenceSummary": {
      "photos": 5200,
      "datasetSizeGb": 72,
      "sourceUri": "archive://in-raj-amber-open/photogrammetry_v2"
    }
  },
  {
    "id": "fort-006",
    "name": "Jaisalmer Fort (Sonar Qila)",
    "category": "INDIAN_FORT",
    "location": "Jaisalmer, Rajasthan",
    "country": "India",
    "scale": "Living fort on Trikuta Hill",
    "complexity": "EXTREME",
    "architectureTypology": "Golden Sandstone Living Citadel",
    "era": "12th Century",
    "heightMeters": 76,
    "footprintSqMeters": 550000,
    "captureTypes": [
      "DRONE_HIGH_RES",
      "STREET_WALKTHROUGH"
    ],
    "expectedImages": 42000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Tri-walled yellow sandstone fortress with 99 bastions, inhabited by 4,000 residents.",
    "keyChallenges": [
      "Identical monochromatic yellow sandstone texturing across thousands of buildings",
      "Extremely narrow 1.5m pedestrian alleys causing acute viewing angles"
    ],
    "benchmarkObjectives": [
      "Low-texture yellow sandstone feature matching in blinding sunlight",
      "Ultra-narrow alley multi-view geometry alignment"
    ]
  },
  {
    "id": "fort-007",
    "name": "Agra Fort",
    "category": "INDIAN_FORT",
    "location": "Agra, Uttar Pradesh",
    "country": "India",
    "scale": "94-acre crescent-shaped citadel",
    "complexity": "EXTREME",
    "architectureTypology": "Imperial Mughal Citadel",
    "era": "16th Century",
    "heightMeters": 70,
    "footprintSqMeters": 380000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 36000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "CAD_SURVEY",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Red sandstone double ramparts with monumental Delhi and Amar Singh gates, Jahangiri Mahal, and Khas Mahal.",
    "keyChallenges": [
      "Massive 21.4m high continuous red sandstone curtain walls",
      "Tourist crowds obstructing ground observation lines of sight"
    ],
    "benchmarkObjectives": [
      "Moving crowd dynamic object removal",
      "Red sandstone relief vs marble inlay boundary alignment"
    ],
    "evidenceSummary": {
      "photos": 4100,
      "datasetSizeGb": 58,
      "sourceUri": "archive://in-up-agrafort/photogrammetry_open"
    }
  },
  {
    "id": "fort-008",
    "name": "Red Fort (Lal Qila)",
    "category": "INDIAN_FORT",
    "location": "Old Delhi",
    "country": "India",
    "scale": "254.67 acres (UNESCO)",
    "complexity": "EXTREME",
    "architectureTypology": "Octagonal Mughal Imperial Citadel",
    "era": "17th Century",
    "heightMeters": 33,
    "footprintSqMeters": 1030000,
    "captureTypes": [
      "DRONE_MAPPING",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 41000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Octagonal red sandstone fortress with 2.41 km perimeter, Lahore Gate, Diwan-i-Aam, and Diwan-i-Khas.",
    "keyChallenges": [
      "High-security urban zone with stringent flight restrictions",
      "Urban smog in Delhi impacting radiometric calibration"
    ],
    "benchmarkObjectives": [
      "Radiometric normalization across variable haze and pollution",
      "Octagonal geometric symmetry constraints in bundle adjustment"
    ]
  },
  {
    "id": "fort-009",
    "name": "Gwalior Fort",
    "category": "INDIAN_FORT",
    "location": "Gwalior, Madhya Pradesh",
    "country": "India",
    "scale": "3 km² basalt hill fortress",
    "complexity": "EXTREME",
    "architectureTypology": "Precipitous Hill Fortress with Enamelled Tiles",
    "era": "8th - 15th Century",
    "heightMeters": 100,
    "footprintSqMeters": 3000000,
    "captureTypes": [
      "DRONE_OBLIQUE",
      "ROCK_FACE_SCAN"
    ],
    "expectedImages": 39000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Man Mandir Palace with yellow, blue, and green enamelled ceramic duck/elephant tiles and rock-cut Jain colossi.",
    "keyChallenges": [
      "100m vertical sandstone cliffs with colossal rock-cut statues",
      "Vivid glazed ceramic tile vs matte sandstone lighting disparity"
    ],
    "benchmarkObjectives": [
      "Vertical rock-face Jain colossi multi-scale mesh generation",
      "Enamelled tile color preservation"
    ]
  },
  {
    "id": "fort-010",
    "name": "Golconda Fort",
    "category": "INDIAN_FORT",
    "location": "Hyderabad, Telangana",
    "country": "India",
    "scale": "11 km outer wall, 120m granite hill",
    "complexity": "EXTREME",
    "architectureTypology": "Acoustic Concentric Granite Citadel",
    "era": "12th - 16th Century",
    "heightMeters": 120,
    "footprintSqMeters": 4000000,
    "captureTypes": [
      "DRONE_GRID",
      "HANDHELD_VIDEO"
    ],
    "expectedImages": 44000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Concentric granite battlements with famous acoustic signaling domes (Fateh Darwaza to Baradari).",
    "keyChallenges": [
      "Enormous weathered granite boulders integrated directly into ramparts",
      "Subterranean acoustic chamber geometry"
    ],
    "benchmarkObjectives": [
      "Natural boulder geology vs man-made masonry segmentation",
      "Acoustic chamber internal spatial volume extraction"
    ]
  },
  {
    "id": "wond-001",
    "name": "Great Pyramid of Giza",
    "category": "WORLD_WONDER",
    "location": "Giza Plateau",
    "country": "Egypt",
    "scale": "138.8 m height, 230.3 m base",
    "complexity": "EXTREME",
    "architectureTypology": "Step-core Monumental Masonry",
    "era": "c. 2560 BCE",
    "heightMeters": 138.8,
    "footprintSqMeters": 53000,
    "captureTypes": [
      "AERIAL_DRONE",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 34000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "2.3 million limestone blocks laid in 203 surviving courses with subtle concave core indents.",
    "keyChallenges": [
      "Repetitive desert limestone blocks lacking distinct high-frequency texture",
      "Slight 8-sided concave indentation along cardinal faces"
    ],
    "benchmarkObjectives": [
      "Detection of 8-sided concavity anomaly across 230m base",
      "Repetitive block feature disambiguation"
    ],
    "evidenceSummary": {
      "photos": 5800,
      "datasetSizeGb": 82,
      "sourceUri": "archive://eg-giza-open/pyramid_khufu_2023"
    }
  },
  {
    "id": "wond-002",
    "name": "Great Wall of China",
    "category": "WORLD_WONDER",
    "location": "Beijing / Northern Frontier",
    "country": "China",
    "scale": "21,196 km total (Mutianyu Section: 5.4 km)",
    "complexity": "EXTREME",
    "architectureTypology": "Linear Ridge Mountain Rampart",
    "era": "Ming Dynasty (14th - 17th Century)",
    "heightMeters": 8.5,
    "footprintSqMeters": 2500000,
    "captureTypes": [
      "LONG_RANGE_DRONE",
      "LIDAR_UAV"
    ],
    "expectedImages": 48000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "SATELLITE_DEM",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Crenellated brick-and-granite defensive parapet following razor-edge mountain ridges.",
    "keyChallenges": [
      "Extreme linear aspect ratio (length vs width > 10,000:1)",
      "Steep mountain valley shadows and seasonal fog"
    ],
    "benchmarkObjectives": [
      "Ultra-long baseline SLAM loop closure over serpentine ridgelines",
      "Watchtower structural orientation extraction"
    ]
  },
  {
    "id": "wond-003",
    "name": "Petra (Al-Khazneh)",
    "category": "WORLD_WONDER",
    "location": "Ma'an Governorate",
    "country": "Jordan",
    "scale": "43 m height, 30 m width carved facade",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Rock-Cut Hellenistic Nabataean",
    "era": "1st Century CE",
    "heightMeters": 43,
    "footprintSqMeters": 1800,
    "captureTypes": [
      "TERRESTRIAL_LIDAR",
      "GROUND_PHOTOGRAMMETRY"
    ],
    "expectedImages": 19500,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Classical Hellenistic facade carved directly out of red-pink rose sandstone cliff face at Siq gorge.",
    "keyChallenges": [
      "Narrow Siq gorge severely limiting camera baseline angles",
      "Weathered sandstone relief details subject to granular erosion"
    ],
    "benchmarkObjectives": [
      "Rock-face integration with architectural Corinthian capitals",
      "Deeply recessed portico interior illumination balance"
    ],
    "evidenceSummary": {
      "photos": 3400,
      "datasetSizeGb": 44,
      "sourceUri": "archive://jo-petra-open/treasury_siq"
    }
  },
  {
    "id": "wond-004",
    "name": "Colosseum (Flavian Amphitheatre)",
    "category": "WORLD_WONDER",
    "location": "Rome",
    "country": "Italy",
    "scale": "189 m x 156 m, 48.5 m height",
    "complexity": "EXTREME",
    "architectureTypology": "Elliptical Tiered Arcaded Arena",
    "era": "70 - 80 CE",
    "heightMeters": 48.5,
    "footprintSqMeters": 24000,
    "captureTypes": [
      "DRONE_ORBIT",
      "TERRESTRIAL_LIDAR",
      "WALKTHROUGH"
    ],
    "expectedImages": 38000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Four-tiered travertine arcades (Doric, Ionic, Corinthian) with exposed subterranean hypogeum maze.",
    "keyChallenges": [
      "Open-air hypogeum subterranean wall complexity with heavy self-occlusion",
      "Elliptical geometry requiring non-circular parametric fitting"
    ],
    "benchmarkObjectives": [
      "Subterranean hypogeum corridor topological mesh extraction",
      "Elliptical curve parameter extraction from multi-view geometry"
    ],
    "evidenceSummary": {
      "photos": 5100,
      "datasetSizeGb": 69.5,
      "sourceUri": "archive://it-rome-heritage/colosseum_full"
    }
  },
  {
    "id": "wond-005",
    "name": "Chichén Itzá (El Castillo)",
    "category": "WORLD_WONDER",
    "location": "Yucatán",
    "country": "Mexico",
    "scale": "30 m height, 55.3 m square base",
    "complexity": "HIGH",
    "architectureTypology": "Radial Step Pyramid with Temple",
    "era": "c. 9th - 12th Century CE",
    "heightMeters": 30,
    "footprintSqMeters": 3050,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "TERRESTRIAL_SCAN"
    ],
    "expectedImages": 14500,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Four 91-step staircases (365 total) with serpent head balustrades on nine tiered platforms.",
    "keyChallenges": [
      "Four-fold rotational symmetry causing 90-degree pose ambiguities in feature matching",
      "Serpent shadow phenomenon requires exact solar ray calibration"
    ],
    "benchmarkObjectives": [
      "Rotational symmetry disambiguation via unique micro-chipping features",
      "Staircase pitch and riser precision validation"
    ]
  },
  {
    "id": "wond-006",
    "name": "Machu Picchu",
    "category": "WORLD_WONDER",
    "location": "Cusco Region",
    "country": "Peru",
    "scale": "2,430 m elevation Incan citadel",
    "complexity": "EXTREME",
    "architectureTypology": "Ashlar Dry-Stone Terraced Citadel",
    "era": "c. 1450 CE",
    "heightMeters": 2430,
    "footprintSqMeters": 325000,
    "captureTypes": [
      "AERIAL_DRONE_GRID",
      "HANDHELD_RGB"
    ],
    "expectedImages": 46000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "SATELLITE_DEM",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Fitted dry-stone ashlar masonry without mortar on agricultural terraces overlooking Urubamba river canyon.",
    "keyChallenges": [
      "Precipitous 400m sheer Andean mountain drop-offs",
      "Fast-moving mountain mist and cloud cover creating radiometric mismatches"
    ],
    "benchmarkObjectives": [
      "Andean terrain steep-gradient mesh stability",
      "Dry-stone mortarless joint boundary extraction"
    ]
  },
  {
    "id": "wond-007",
    "name": "Taj Mahal",
    "category": "WORLD_WONDER",
    "location": "Agra, Uttar Pradesh",
    "country": "India",
    "scale": "73 m dome height on 95m plinth",
    "complexity": "EXTREME",
    "architectureTypology": "Mughal Bilateral Marble Mausoleum",
    "era": "1632 - 1653 CE",
    "heightMeters": 73,
    "footprintSqMeters": 17000,
    "captureTypes": [
      "DRONE_PERIMETER",
      "TERRESTRIAL_LIDAR",
      "HANDHELD_PHOTO"
    ],
    "expectedImages": 35000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "CAD_SURVEY",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Pure white Makrana marble tomb with four 40m minarets tilted outward 3 degrees, reflecting pool, and charbagh.",
    "keyChallenges": [
      "Translucent white Makrana marble subsurface scattering",
      "Extremely high bilateral symmetry causing camera pose flips",
      "Outward minaret tilt (3°) must not be 'straightened' by verticality priors"
    ],
    "benchmarkObjectives": [
      "Subsurface scattering radiometric compensation on white marble",
      "Verification of deliberate 3° outward minaret tilt",
      "Pietra dura semi-precious stone inlay boundary segmentation"
    ],
    "evidenceSummary": {
      "photos": 6200,
      "datasetSizeGb": 88.4,
      "sourceUri": "archive://in-up-agra-taj/photogrammetry_v1"
    }
  },
  {
    "id": "wond-008",
    "name": "Christ the Redeemer",
    "category": "WORLD_WONDER",
    "location": "Rio de Janeiro",
    "country": "Brazil",
    "scale": "30 m statue on 8m pedestal, 28m arm span",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Art Deco Reinforced Concrete Monolith",
    "era": "1922 - 1931 CE",
    "heightMeters": 38,
    "footprintSqMeters": 450,
    "captureTypes": [
      "DRONE_ORBIT",
      "TERRESTRIAL_PHOTOGRAMMETRY"
    ],
    "expectedImages": 14200,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Reinforced concrete shell covered with 6 million triangular soapstone tiles on 710m Corcovado summit.",
    "keyChallenges": [
      "710m sheer vertical drop beneath pedestal with zero ground sensor access",
      "6 million triangular soapstone tiles creating repetitive fine mosaic artifacts"
    ],
    "benchmarkObjectives": [
      "High-angle drone orbit alignment over sheer precipice",
      "Soapstone mosaic smoothing without losing facial feature contours"
    ]
  },
  {
    "id": "wond-009",
    "name": "Angkor Wat",
    "category": "WORLD_WONDER",
    "location": "Siem Reap",
    "country": "Cambodia",
    "scale": "162.6 hectares (largest religious structure)",
    "complexity": "EXTREME",
    "architectureTypology": "Khmer Temple Mountain",
    "era": "12th Century CE",
    "heightMeters": 65,
    "footprintSqMeters": 1626000,
    "captureTypes": [
      "DRONE_GRID",
      "TERRESTRIAL_LIDAR",
      "WALKTHROUGH_RGB"
    ],
    "expectedImages": 54000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "SATELLITE_DEM",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Sandstone quincunx of lotus-bud towers with 800m of continuous bas-relief galleries and 190m moat.",
    "keyChallenges": [
      "Immense spatial extent (1.6 km²)",
      "Deep shaded gallery bas-reliefs under heavy stone eaves with high dynamic range"
    ],
    "benchmarkObjectives": [
      "Deep gallery shadow detail recovery under extreme dynamic range (14+ EV)",
      "Lotus-bud tower spire multi-view alignment"
    ]
  },
  {
    "id": "wond-010",
    "name": "Hagia Sophia",
    "category": "WORLD_WONDER",
    "location": "Istanbul",
    "country": "Turkey",
    "scale": "55.6 m dome height, 31.87 m diameter",
    "complexity": "EXTREME",
    "architectureTypology": "Byzantine Domed Basilica",
    "era": "537 CE",
    "heightMeters": 55.6,
    "footprintSqMeters": 7570,
    "captureTypes": [
      "DRONE_INTERNAL",
      "TERRESTRIAL_LIDAR",
      "HIGH_DYNAMIC_RANGE"
    ],
    "expectedImages": 37400,
    "reconstructionStatus": "PARTIAL",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "Pendentive-supported central dome flanked by two semi-domes, marble revetments, and gold mosaics.",
    "keyChallenges": [
      "Complex pendentive interior curvature with gold mosaic specularity",
      "Extreme lighting contrast between 40 high dome windows and dark nave vaults"
    ],
    "benchmarkObjectives": [
      "Spherical pendentive vault mathematical surface extraction",
      "Gold glass tesserae specular highlight removal"
    ],
    "evidenceSummary": {
      "photos": 4800,
      "datasetSizeGb": 74,
      "sourceUri": "archive://tr-istanbul-hagiasophia/internal_lidar"
    }
  },
  {
    "id": "euro-001",
    "name": "Notre-Dame de Paris",
    "category": "EUROPEAN_HERITAGE",
    "location": "Paris",
    "country": "France",
    "scale": "Early Gothic (69m towers, 96m spire)",
    "complexity": "EXTREME",
    "architectureTypology": "Early French Gothic Cathedral",
    "era": "1163 - 1345 CE",
    "heightMeters": 96,
    "footprintSqMeters": 4800,
    "captureTypes": [
      "POST_FIRE_LIDAR",
      "DRONE_FACADE",
      "INTERIOR_SPHERICAL"
    ],
    "expectedImages": 46000,
    "reconstructionStatus": "EVIDENCE_AVAILABLE",
    "captureAvailability": "AERIAL_ARCHIVE",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "PARTIAL",
    "structuralNotes": "High-soaring flying buttresses, gargoyles, stained-glass rose windows, and post-fire timber framework.",
    "keyChallenges": [
      "Massive scaffolding structures causing severe line-of-sight occlusion",
      "Delicate stone tracery in rose windows (< 15mm cross-section)"
    ],
    "benchmarkObjectives": [
      "Scaffolding occlusion separation from historical limestone masonry",
      "Flying buttress geometric load-path verification"
    ],
    "evidenceSummary": {
      "photos": 7200,
      "lidarScans": 48,
      "datasetSizeGb": 112,
      "sourceUri": "archive://fr-paris-notredame/cnrs_post_fire_2022"
    }
  },
  {
    "id": "euro-002",
    "name": "Cologne Cathedral",
    "category": "EUROPEAN_HERITAGE",
    "location": "Cologne",
    "country": "Germany",
    "scale": "High Gothic (157.38 m twin spires)",
    "complexity": "EXTREME",
    "architectureTypology": "High Gothic Twin-Spire Cathedral",
    "era": "1248 - 1880 CE",
    "heightMeters": 157.4,
    "footprintSqMeters": 7914,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 39500,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Largest twin-spire church in the world with delicate pierced stone openwork spires.",
    "keyChallenges": [
      "Openwork pierced stone tracery allows background sky through spires",
      "Darkened weathered sandstone creates low radiometric contrast"
    ],
    "benchmarkObjectives": [
      "Pierced openwork spire depth separation against sky background",
      "Twin spire vertical alignment over 157m"
    ]
  },
  {
    "id": "euro-003",
    "name": "St. Peter's Basilica",
    "category": "EUROPEAN_HERITAGE",
    "location": "Vatican City",
    "country": "Vatican City",
    "scale": "136.6 m dome height, 220m length",
    "complexity": "EXTREME",
    "architectureTypology": "High Renaissance & Baroque Basilica",
    "era": "1506 - 1626 CE",
    "heightMeters": 136.6,
    "footprintSqMeters": 23000,
    "captureTypes": [
      "INTERIOR_LIDAR",
      "AERIAL_PHOTOGRAMMETRY"
    ],
    "expectedImages": 44000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Michelangelo's monumental dome, Maderno's facade, Bernini's baldachin, and St. Peter's Square colonnade.",
    "keyChallenges": [
      "Massive indoor scale (136m interior vertical clearance)",
      "Polished marble flooring reflections"
    ],
    "benchmarkObjectives": [
      "Interior vault acoustic & spatial dimensioning",
      "Colonnade elliptical layout parameter extraction"
    ]
  },
  {
    "id": "euro-004",
    "name": "Florence Cathedral (Santa Maria del Fiore)",
    "category": "EUROPEAN_HERITAGE",
    "location": "Florence",
    "country": "Italy",
    "scale": "114.5 m dome height, 45m dome span",
    "complexity": "EXTREME",
    "architectureTypology": "Italian Gothic & Early Renaissance",
    "era": "1296 - 1436 CE",
    "heightMeters": 114.5,
    "footprintSqMeters": 8300,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "INTERNAL_LIDAR"
    ],
    "expectedImages": 36000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Brunelleschi's self-supporting octagonal brick double-dome built without wooden centering.",
    "keyChallenges": [
      "Inner and outer double-shell dome clearance with narrow maintenance passageway",
      "Polychrome marble geometric facade patterns (green, white, pink)"
    ],
    "benchmarkObjectives": [
      "Brunelleschi double-dome inner/outer shell thickness measurement",
      "Polychrome geometric marble facade planar alignment"
    ]
  },
  {
    "id": "euro-005",
    "name": "Palace of Versailles",
    "category": "EUROPEAN_HERITAGE",
    "location": "Versailles",
    "country": "France",
    "scale": "67,000 m² palace on 800-hectare estate",
    "complexity": "EXTREME",
    "architectureTypology": "French Baroque Royal Palace",
    "era": "1661 - 1715 CE",
    "heightMeters": 25,
    "footprintSqMeters": 67000,
    "captureTypes": [
      "DRONE_MAPPING",
      "INTERIOR_SLAM"
    ],
    "expectedImages": 49000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Hall of Mirrors (73m), Royal Opera, marble courtyard, and monumental formal parterres and canals.",
    "keyChallenges": [
      "Hall of Mirrors: 357 continuous arched mirrors reflecting opposite windows",
      "Vast 800-hectare parterre garden spatial scale"
    ],
    "benchmarkObjectives": [
      "Hall of Mirrors multi-view specular reflection elimination",
      "Parterre geometric symmetry extraction"
    ]
  },
  {
    "id": "euro-006",
    "name": "Schönbrunn Palace",
    "category": "EUROPEAN_HERITAGE",
    "location": "Vienna",
    "country": "Austria",
    "scale": "1,441 rooms Rococo summer residence",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Habsburg Baroque / Rococo Palace",
    "era": "1740s CE",
    "heightMeters": 32,
    "footprintSqMeters": 45000,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "GARDEN_SURVEY"
    ],
    "expectedImages": 29000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Monumental Habsburg Baroque facade, Gloriette colonnade, and Great Parterre.",
    "keyChallenges": [
      "Monochromatic imperial yellow facade with subtle relief",
      "Gloriette hilltop distance from main palace"
    ],
    "benchmarkObjectives": [
      "Long-distance topological link between main palace and Gloriette"
    ]
  },
  {
    "id": "euro-007",
    "name": "Sagrada Família",
    "category": "EUROPEAN_HERITAGE",
    "location": "Barcelona",
    "country": "Spain",
    "scale": "172.5 m Christ Tower height",
    "complexity": "EXTREME",
    "architectureTypology": "Catalan Modernisme & Organic Gothic",
    "era": "1882 - Present",
    "heightMeters": 172.5,
    "footprintSqMeters": 4500,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "SPHERICAL_LIDAR",
      "ROBOTIC_RGB"
    ],
    "expectedImages": 58000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Gaudí's naturalistic ruled surfaces (hyperboloids, paraboloids, helicoids) and forest-like branching columns.",
    "keyChallenges": [
      "Non-Euclidean organic ruled surfaces that break traditional Manhattan-world planar assumptions",
      "Active construction cranes, safety nets, and scaffolds"
    ],
    "benchmarkObjectives": [
      "Non-planar ruled surface fitting (hyperbolic paraboloids)",
      "Branching tree-column load tree structural topology extraction"
    ]
  },
  {
    "id": "euro-008",
    "name": "Milan Cathedral (Duomo di Milano)",
    "category": "EUROPEAN_HERITAGE",
    "location": "Milan",
    "country": "Italy",
    "scale": "135 marble spires, 3,400 statues",
    "complexity": "EXTREME",
    "architectureTypology": "Italian Rayonnant-Flamboyant Gothic",
    "era": "1386 - 1965 CE",
    "heightMeters": 108.5,
    "footprintSqMeters": 11700,
    "captureTypes": [
      "DRONE_PHOTOGRAMMETRY",
      "TERRESTRIAL_LIDAR"
    ],
    "expectedImages": 45000,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Candoglia pink-white marble exterior with 135 forest-like spires, gargoyles, and rooftop walking terrace.",
    "keyChallenges": [
      "Mutual occlusion among 135 pierced marble spires on the roof terrace",
      "Translucent Candoglia pink marble light transmission"
    ],
    "benchmarkObjectives": [
      "Pierced spire forest occlusion disentanglement",
      "Rooftop open marble tile walkway geometric model"
    ]
  },
  {
    "id": "euro-009",
    "name": "Victoria Memorial",
    "category": "EUROPEAN_HERITAGE",
    "location": "Kolkata, West Bengal",
    "country": "India",
    "scale": "56 m dome height on 26-hectare gardens",
    "complexity": "VERY_HIGH",
    "architectureTypology": "Indo-Saracenic & British Classical",
    "era": "1906 - 1921 CE",
    "heightMeters": 56,
    "footprintSqMeters": 8000,
    "captureTypes": [
      "DRONE_ORBIT",
      "TERRESTRIAL_LIDAR",
      "HANDHELD_PHONE"
    ],
    "expectedImages": 2850,
    "reconstructionStatus": "VALIDATED",
    "captureAvailability": "LOCAL_DATASET",
    "groundTruthAvailability": "TERRESTRIAL_LIDAR",
    "worldIRStatus": "COMPILED",
    "structuralNotes": "Makrana marble monument blending British Classical, Venetian, Egyptian, and Mughal architectural elements with 16-ft bronze Angel of Victory.",
    "keyChallenges": [
      "Translucent Makrana marble dome specular highlights",
      "Southern reflecting pond water reflections",
      "Monsoon tropical humidity atmospheric scattering"
    ],
    "benchmarkObjectives": [
      "High-precision verification of rotunda dome diameter (21.4m ± 0.04m)",
      "Colonnaded plinth gallery structural extraction",
      "Evidence traceability from WorldIR entities to raw calibrated camera rays"
    ],
    "evidenceSummary": {
      "photos": 2850,
      "lidarScans": 12,
      "gcpCount": 8,
      "datasetSizeGb": 38.6,
      "sourceUri": "datasets/victoria_memorial_kolkata/production"
    },
    "metrics": {
      "registeredCameras": 2842,
      "sparsePoints": 482100,
      "densePoints": 18450000,
      "reprojectionErrorPx": 0.42,
      "meanResidualMm": 12.4,
      "uncertaintyMm": 38,
      "detectedObjects": 14,
      "detectedPlanes": 68,
      "detectedRooms": 8,
      "processingTimeMin": 42.5
    }
  },
  {
    "id": "euro-010",
    "name": "Chhatrapati Shivaji Maharaj Terminus (CSMT)",
    "category": "EUROPEAN_HERITAGE",
    "location": "Mumbai, Maharashtra",
    "country": "India",
    "scale": "Victorian Italianate Gothic (UNESCO)",
    "complexity": "EXTREME",
    "architectureTypology": "Victorian High Gothic Revival with Indian Palace Details",
    "era": "1878 - 1887 CE",
    "heightMeters": 45,
    "footprintSqMeters": 14000,
    "captureTypes": [
      "DRONE",
      "STATIONARY_LIDAR",
      "WALKTHROUGH_VIDEO"
    ],
    "expectedImages": 33800,
    "reconstructionStatus": "NOT_STARTED",
    "captureAvailability": "NOT_YET_CAPTURED",
    "groundTruthAvailability": "NOT_AVAILABLE",
    "worldIRStatus": "PENDING",
    "structuralNotes": "Victorian Italianate Gothic revival with Indian palace elements, colossal stone dome, turrets, and pointed arches.",
    "keyChallenges": [
      "Dense 24/7 commuter traffic occlusion around all entrances",
      "Mixed materials (sandstone, granite, Italian marble) with disparate albedos"
    ],
    "benchmarkObjectives": [
      "Dynamic commuter pedestrian removal from facade baseline",
      "Carved stone peacock and lion ornamental relief extraction"
    ]
  }
];
