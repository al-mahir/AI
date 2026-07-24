import { useEffect, useMemo, useState } from "react";
import { CloseIcon } from "./Icons";
import { defaultConfig, loadMoshafSchema } from "../lib/moshaf";
import type { MistakeLog, MoshafConfig, MoshafField, SuraInfo } from "../lib/types";

/** The sūra index: the only way to move around the muṣḥaf, so it is also the search. */
export function IndexSheet({
  suras,
  onPick,
  onClose,
}: {
  suras: SuraInfo[];
  onPick: (sura: SuraInfo) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState("");

  const hits = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return suras;
    // Match the Arabic name, the transliteration, or the number — a reciter reaches
    // for whichever they know.
    const bare = stripHarakat(needle);
    return suras.filter(
      (s) =>
        stripHarakat(s.name_ar).includes(bare) ||
        s.name_en.toLowerCase().includes(needle) ||
        String(s.sura) === needle,
    );
  }, [q, suras]);

  return (
    <div className="sheet" role="dialog" aria-label="فهرس السور">
      <div className="sheet__bar">
        <button className="iconbtn" onClick={onClose} title="إغلاق">
          <CloseIcon />
        </button>
        <input
          className="search"
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="ابحث عن سورة…"
          aria-label="ابحث عن سورة"
        />
      </div>
      <div className="sheet__body">
        {hits.map((s) => (
          <button key={s.sura} className="surarow" onClick={() => onPick(s)}>
            <span className="surarow__no">{s.sura}</span>
            <span>
              <div className="surarow__name">{s.name_ar}</div>
              <div className="surarow__meta">
                {s.name_en} · {s.ayat} آية · {s.revelation === "makkah" ? "مكية" : "مدنية"}
              </div>
            </span>
            <span className="surarow__spacer" />
            <span className="surarow__meta">ص {s.page}</span>
          </button>
        ))}
        {!hits.length && <p className="empty">لا سورة بهذا الاسم.</p>}
      </div>
    </div>
  );
}

const stripHarakat = (s: string) => s.replace(/[ً-ْـٰ]/g, "");

/** The session's mistake log — the "past mistakes" the brain icon opens. */
export function MistakesSheet({
  log,
  suras,
  onClose,
}: {
  log: MistakeLog[];
  suras: SuraInfo[];
  onClose: () => void;
}) {
  const name = (n: number) => suras.find((s) => s.sura === n)?.name_ar ?? n;
  const errors = log.filter((m) => m.status === "error");

  return (
    <div className="sheet" role="dialog" aria-label="الأخطاء">
      <div className="sheet__bar">
        <button className="iconbtn" onClick={onClose} title="إغلاق">
          <CloseIcon />
        </button>
        <span className="sheet__title">أخطاء الجلسة</span>
        <span className="surarow__spacer" />
        <span className="surarow__meta">
          {errors.length} خطأ · {log.length - errors.length} تنبيه
        </span>
      </div>
      <div className="sheet__body">
        {log.length === 0 ? (
          <p className="empty">
            لا أخطاء بعد.
            <br />
            ابدأ التلاوة وسيظهر هنا كل ما تُنبَّه عليه.
          </p>
        ) : (
          [...log].reverse().map((m, i) => (
            <div className="mistake" key={`${m.sura}:${m.aya}:${m.word_idx}:${i}`}>
              <span className="mistake__word">{m.uthmani}</span>
              <span className="mistake__where">
                {name(m.sura)} {m.aya}:{m.word_idx + 1}
              </span>
              <span className={`mistake__kind mistake__kind--${m.status}`}>
                {m.status === "error" ? m.rules[0] ?? kindLabel(m.kinds[0]) : "غير مؤكَّد"}
              </span>
            </div>
          ))
        )}
        {/* `almost` is a hint, not a mistake — say so where it is being counted. */}
        {log.some((m) => m.status === "almost") && (
          <p className="empty" style={{ padding: "1.25rem 0.5rem 0", textAlign: "start" }}>
            «غير مؤكَّد» ليس خطأً: لم يسمعك النموذج بوضوح كافٍ ليحكم.
          </p>
        )}
      </div>
    </div>
  );
}

const kindLabel = (k?: string) =>
  k === "tajweed" ? "تجويد" : k === "tashkeel" ? "تشكيل" : k === "sifa" ? "صفة" : "نطق";

/**
 * The reciter's moshaf attributes — madd holding lengths and the rule choices that
 * change what a correct recitation sounds like, so scoring matches how THIS reciter
 * reads. Generated from the backend schema; a change applies to the next session.
 */
export function MoshafSheet({
  value,
  onSave,
  onClose,
}: {
  value: MoshafConfig | null;
  onSave: (cfg: MoshafConfig | null) => void;
  onClose: () => void;
}) {
  const [fields, setFields] = useState<MoshafField[] | null>(null);
  const [error, setError] = useState(false);
  const [cfg, setCfg] = useState<MoshafConfig>({});

  useEffect(() => {
    let live = true;
    loadMoshafSchema()
      .then((fs) => {
        if (!live) return;
        setFields(fs);
        setCfg(value ?? defaultConfig(fs));
      })
      .catch(() => live && setError(true));
    return () => {
      live = false;
    };
  }, [value]);

  const set = (key: string, v: string | number) => setCfg((c) => ({ ...c, [key]: v }));

  const apply = () => {
    onSave(cfg);
    onClose();
  };
  const useModelDefault = () => {
    onSave(null); // send no moshaf — the backend falls back to its own default
    onClose();
  };
  const resetShown = () => fields && setCfg(defaultConfig(fields));

  return (
    <div className="sheet" role="dialog" aria-label="إعدادات التلاوة">
      <div className="sheet__bar">
        <button className="iconbtn" onClick={onClose} title="إغلاق">
          <CloseIcon />
        </button>
        <span className="sheet__title">خصائص المصحف والتلاوة</span>
        <span className="surarow__spacer" />
        {fields && (
          <button className="moshaf__link" onClick={resetShown}>
            القيم الافتراضية
          </button>
        )}
      </div>

      <div className="sheet__body">
        {error ? (
          <p className="empty">تعذّر تحميل الإعدادات — تأكّد من تشغيل خدمة التلاوة.</p>
        ) : !fields ? (
          <p className="empty">يُحمّل…</p>
        ) : (
          <>
            <p className="moshaf__hint">
              تُطبَّق هذه الخصائص على الجلسة التالية، وتُضبط بها مطابقة التجويد لطريقة قراءتك.
            </p>
            {fields.map((f) => (
              <div className="moshaf__field" key={f.key}>
                <label className="moshaf__label" title={f.description ?? undefined}>
                  {f.name_ar}
                </label>
                <div className="moshaf__opts" role="group" aria-label={f.name_ar}>
                  {f.options.map((o) => (
                    <button
                      key={String(o.value)}
                      className={`moshaf__opt${cfg[f.key] === o.value ? " moshaf__opt--on" : ""}`}
                      aria-pressed={cfg[f.key] === o.value}
                      onClick={() => set(f.key, o.value)}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </>
        )}
      </div>

      {fields && !error && (
        <div className="sheet__foot">
          <button className="moshaf__ghost" onClick={useModelDefault}>
            الوضع الافتراضي
          </button>
          <button className="moshaf__save" onClick={apply}>
            حفظ
          </button>
        </div>
      )}
    </div>
  );
}
