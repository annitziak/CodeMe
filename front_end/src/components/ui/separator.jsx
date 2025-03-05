import * as React from "react";
import * as SeparatorPrimitive from "@radix-ui/react-separator";

import { cn } from "@/lib/utils";

function Separator({
  className,
  orientation = "horizontal",
  decorative = true,
  ...props
}) {
  return (
    <SeparatorPrimitive.Root
      data-slot="separator-root"
      decorative={decorative}
      orientation={orientation}
      className={cn(
        "bg-gray-400 shrink-0 opacity-80",
        "data-[orientation=horizontal]:h-[1px] data-[orientation=horizontal]:w-full",
        "data-[orientation=vertical]:h-full data-[orientation=vertical]:w-[1px]",
        "my-2", // Small margin to keep it subtle but distinct
        className
      )}
      {...props}
    />
  );
}

export { Separator };
