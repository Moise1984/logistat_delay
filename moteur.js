// Moteur LogiStat (version web) : refait exactement le calcul de l'application Python
// avec le modèle final du bloc 3 (LightGBM) : champs déduits, encodage one-hot,
// parcours des 299 arbres, puis explication des facteurs.
const SEUIL_NEGLIGEABLE = 0.01;
const DERIVEES_DATE = ["heure_commande", "jour_semaine", "mois_commande"];

function creerMoteur(M, ALIAS) {
  const P = M.pipeline;
  // position de chaque modalité dans l'encodage one-hot (catégorielles d'abord)
  const debut = [];
  let n = 0;
  P.categories.forEach((cats) => { debut.push(n); n += cats.length; });
  const positions = P.categories.map((cats) => new Map(cats.map((c, i) => [c, i])));
  const vide = (v) => v === undefined || v === null || (typeof v === "number" && Number.isNaN(v)) || String(v).trim() === "";

  // Champs facultatifs absents → valeurs par défaut (ModeleBloc3.completer)
  function completer(cmd) {
    const c = { ...cmd };
    if (!vide(c.categorie) && !("departement" in c)) c.departement = M.departement_par_categorie[c.categorie];
    for (const v of M.variables) if (vide(c[v]) && v in M.defauts) c[v] = M.defauts[v];
    return c;
  }

  // Champs LogiStat → les 15 variables du bloc 3 (ModeleBloc3.traduire), puis encodage
  // one-hot identique à scikit-learn (modalité inconnue : que des zéros)
  function vecteur(cmd) {
    const c = completer(cmd), x = {};
    for (const [nous, eux] of Object.entries(M.correspondance)) x[eux] = c[nous];
    x["Days for shipment (scheduled)"] = M.delai_par_mode[c.mode_expedition];
    x["Market"] = M.marche_par_region[c.region_commande];
    Object.assign(x, M.fixes);
    const X = new Array(n).fill(0);
    P.categorielles.forEach((k, i) => {
      const j = vide(x[k]) ? undefined : positions[i].get(String(x[k]));
      if (j !== undefined) X[debut[i] + j] = 1;
    });
    P.numeriques.forEach((k) => X.push(vide(x[k]) ? NaN : Number(x[k])));
    return X;
  }

  // Parcours d'un arbre LightGBM : coupure « ≤ seuil » ; valeur manquante traitée comme 0
  function arbre(a, X) {
    let i = 0;
    while (a.f[i] >= 0) {
      let val = X[a.f[i]];
      if (Number.isNaN(val)) val = 0;
      i = val <= a.s[i] ? a.g[i] : a.d[i];
    }
    return a.v[i];
  }

  function probabilite(cmd) {
    const X = vecteur(cmd);
    let z = 0;
    for (const a of P.arbres) z += arbre(a, X);
    return 1 / (1 + Math.exp(-P.sigmoide * z));
  }

  // Effet d'un facteur = P(commande) − moyenne des P(commande avec ce facteur remplacé)
  function facteurs(cmd) {
    const base = completer(cmd), p = probabilite(base);
    const dansGroupe = new Set(Object.values(M.groupes).flat());
    const liste = Object.entries(M.groupes).concat(M.variables.filter((v) => !dansGroupe.has(v)).map((v) => [v, [v]]));
    const out = liste.map(([nom, vars]) => {
      let somme = 0;
      for (const ref of M.comparaison) {
        const variante = { ...base };
        vars.forEach((v) => (variante[v] = ref[v]));
        somme += probabilite(variante);
      }
      const effet = p - somme / M.comparaison.length;
      const sens = effet >= SEUIL_NEGLIGEABLE ? "augmente le risque" : effet <= -SEUIL_NEGLIGEABLE ? "réduit le risque" : "effet négligeable";
      return { variable: nom, valeur: vars.map((v) => base[v]).join(" · "), effet, sens };
    });
    out.sort((a, b) => b.effet - a.effet);
    return { probabilite: p, facteurs: out };
  }

  // ---------- import de fichier (donnees/import_utilisateur.py)
  function lireDate(v) {
    if (v instanceof Date) return { iso: v.toISOString().slice(0, 16).replace("T", " "), y: v.getFullYear(), mo: v.getMonth() + 1, d: v.getDate(), h: v.getHours() };
    const s = String(v).trim();
    let r = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?/);
    if (r) return { iso: s, y: +r[1], mo: +r[2], d: +r[3], h: r[4] === undefined ? null : +r[4] };
    r = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})(?:\s+(\d{1,2}):(\d{2}))?/);
    if (r) return { iso: s, y: +r[3], mo: +r[1], d: +r[2], h: r[4] === undefined ? null : +r[4] };
    return null;
  }

  function normaliser(lignes) {
    let rows = lignes.map((r) => {
      const o = {};
      for (const [k, v] of Object.entries(r)) {
        const nom = ALIAS[String(k).trim().toLowerCase()] || String(k).trim();
        if (!(nom in o)) o[nom] = typeof v === "string" ? v.trim() : v;
      }
      return o;
    });
    const presentes = new Set(rows.flatMap((o) => Object.keys(o)));
    rows.forEach((o) => {
      if (!vide(o.date_commande)) {
        const d = lireDate(o.date_commande);
        if (d) {
          o.date_commande = d.iso;
          const jour = (new Date(Date.UTC(d.y, d.mo - 1, d.d)).getUTCDay() + 6) % 7;
          if (!presentes.has("heure_commande") && d.h !== null) o.heure_commande = d.h;
          if (!presentes.has("jour_semaine")) o.jour_semaine = jour;
          if (!presentes.has("mois_commande")) o.mois_commande = d.mo;
        }
      }
      DERIVEES_DATE.forEach((k) => { if (k in o && !vide(o[k])) o[k] = Number(o[k]); });
      if (vide(o.region_commande) && !vide(o.pays_commande)) o.region_commande = M.pays_vers_region[o.pays_commande];
      if ("en_retard" in o) {
        const t = String(o.en_retard).trim().toLowerCase();
        o.en_retard = ["1", "true", "vrai", "oui"].includes(t) ? true : ["0", "false", "faux", "non"].includes(t) ? false : null;
      }
      if ("statut_livraison" in o) o.annulee = o.statut_livraison === "Shipping canceled";
    });
    if (rows.length && "id_commande" in rows[0]) {
      const vus = new Map();
      rows.forEach((o) => { if (!vus.has(String(o.id_commande))) vus.set(String(o.id_commande), o); });
      rows = [...vus.values()];
    }
    return rows;
  }

  const manquantes = (rows) => M.obligatoires.filter((v) => !rows.some((o) => !vide(o[v])));

  // Mêmes avertissements que import_utilisateur.controler()
  function controler(rows) {
    const msg = [], n = rows.length;
    for (const v of M.obligatoires) {
      const manq = rows.filter((o) => vide(o[v])).length;
      if (manq) msg.push(`${v} : ${manq} valeur(s) manquante(s) sur ${n}, remplacée(s) par la valeur la plus fréquente.`);
      if (M.modalites[v]) {
        const connues = new Set(M.modalites[v]);
        const inc = rows.map((o) => o[v]).filter((x) => !vide(x) && !connues.has(String(x)));
        if (inc.length) msg.push(`${v} : ${inc.length} valeur(s) jamais vue(s) à l'entraînement (ex. ${[...new Set(inc.map(String))].sort().slice(0, 3).join(", ")}) ; le modèle n'en tient pas compte.`);
      }
    }
    const hors = rows.filter((o) => typeof o.heure_commande === "number" && (o.heure_commande < 0 || o.heure_commande > 23)).length;
    if (hors) msg.push(`heure_commande : ${hors} valeur(s) hors de 0 à 23.`);
    const ann = rows.filter((o) => o.annulee).length;
    if (ann) msg.push(`${ann} commande(s) annulée(s) : elles ne seront jamais livrées, elles sont exclues de la comparaison avec la réalité.`);
    return msg;
  }

  return { completer, probabilite, facteurs, normaliser, manquantes, controler };
}
if (typeof module !== "undefined") module.exports = { creerMoteur };
